# Copyright (c) Facebook, Inc. and its affiliates.
import os
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from .third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from .third_party.pointnet2.pointnet2_utils import furthest_point_sample
from .utils.pc_util import scale_points, shift_scale_points

from .helpers import GenericMLP
from .position_embedding import PositionEmbeddingCoordsSine
from .transformer import (
    MaskedTransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .epcl import CLIPVITEncoder, TaskEmbEncoder


class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model3DETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        dataset_name,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        use_task_emb=True,
        encoder_only=False,
        vit_only=False,
    ):
        super().__init__()
        # ====================
        # task embedding
        self.use_task_emb = use_task_emb
        if use_task_emb:
            if dataset_name == "scannet":
                num_cls = 19
            else:
                raise NotImplementedError

        if self.use_task_emb:
            self.te_tok = torch.arange(num_cls).long()
            self.te_encoder = TaskEmbEncoder(token_num=num_cls, emb_dim=encoder_dim)
        # ====================
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)
        self.vit_only = vit_only
        self.encoder_only = encoder_only
        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)

    def get_prompt(self, batch_size, te_token, te_encoder, device):
        prompts_tokens = te_token.expand(batch_size, -1).view(batch_size, -1).to(device)
        past_key_values = te_encoder(prompts_tokens)

        return past_key_values

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        if self.pre_encoder is None:
            return self.encoder(point_clouds)
        xyz, features = self._break_up_pc(
            point_clouds
        )  # max-pooling feature of each group
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(0, 2, 1)

        if self.use_task_emb:
            task_emb = self.get_prompt(
                batch_size=pre_enc_features.shape[0],
                te_token=self.te_tok,
                te_encoder=self.te_encoder,
                device=pre_enc_features.device,
            )
        else:
            task_emb = None

        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz, task_emb=task_emb
        )

        return enc_xyz, enc_features, enc_inds

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False, vit_only=True):
        point_clouds = inputs["point_clouds"]
        enc_xyz, enc_features, enc_inds = self.run_encoder(
            point_clouds
        )  # group center, group feature and group center point in ori point cloud

        if vit_only or self.vit_only:
            return enc_xyz, enc_features

        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(0, 2, 1)
        ).permute(2, 0, 1)

        if encoder_only or self.encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(
            enc_xyz, point_cloud_dims
        )  # FPS from enc_xyz to get the queries, and project query centers to high dimension features
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1).contiguous()
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[
            0
        ]  # [num_layer, N, B, dim]

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        return box_predictions


class Model3DETR_encoder(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        dataset_name="scannet",
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        num_queries=256,
        use_task_emb=True,
        encoder_only=False,
        vit_only=False,
    ):
        super().__init__()
        # ====================
        self.use_task_emb = use_task_emb
        if use_task_emb:
            if dataset_name == "scannet":
                num_cls = 19
            else:
                raise NotImplementedError

        if self.use_task_emb:
            self.te_tok = torch.arange(num_cls).long()
            self.te_encoder = TaskEmbEncoder(token_num=num_cls, emb_dim=encoder_dim)
        # ====================
        self.pre_encoder = pre_encoder
        self.encoder = encoder

        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.num_cls = num_cls
        self.vit_only = vit_only
        self.encoder_only = encoder_only
        self.num_queries = num_queries

    def get_prompt(self, batch_size, te_token, te_encoder, device):
        prompts_tokens = te_token.expand(batch_size, -1).view(batch_size, -1).to(device)
        past_key_values = te_encoder(prompts_tokens)

        return past_key_values

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        if self.pre_encoder is None:
            return self.encoder(point_clouds)
        xyz, features = self._break_up_pc(
            point_clouds
        )  # max-pooling feature of each group; coordinate & color
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)

        pre_enc_features = pre_enc_features.permute(0, 2, 1)

        if self.use_task_emb:
            task_emb = self.get_prompt(
                batch_size=pre_enc_features.shape[0],
                te_token=self.te_tok,
                te_encoder=self.te_encoder,
                device=pre_enc_features.device,
            )
        else:
            task_emb = None

        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz, task_emb=task_emb
        )

        return enc_xyz, enc_features, enc_inds

    def forward(self, point_clouds):
        enc_xyz, enc_features, enc_inds = self.run_encoder(
            point_clouds
        )  # group center, group feature and group center point in ori point cloud
        return enc_xyz, enc_features


class Pointnet2Preencoder(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        scale_ratio = (
            args.preenc_npoints // args.vit_num_token
        )  # 2048 // (256 or 512 or 1024) - 8 or 4 or 2
        # TODO: adaptive to input arguments
        # use 2 layer to downsample 2048 token to target number; ! too many layers maybe difficult to train
        self.layers = nn.ModuleList(
            [
                # 2048, 256
                PointnetSAModuleVotes(
                    radius=0.2,
                    nsample=64,
                    npoint=args.preenc_npoints,
                    mlp=[input_dim, 64, 256],
                    normalize_xyz=True,
                ),
                # num_token, 1024
                PointnetSAModuleVotes(
                    radius=0.2 * scale_ratio,
                    nsample=64 * scale_ratio,
                    npoint=args.vit_num_token,
                    mlp=[256, 512, args.enc_dim],
                    normalize_xyz=True,
                ),
            ]
        )

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor = None,
        inds: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        for i in range(len(self.layers)):
            if features is not None:
                features = features.type(torch.float32)
            xyz, features, inds = self.layers[i](xyz, features)
        return xyz, features, inds


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    # hierarchical downsample
    if args.pointnet_downsample:
        preencoder = Pointnet2Preencoder(args, 3 * int(args.use_color))
    else:
        preencoder = PointnetSAModuleVotes(
            radius=args.preenc_radius,
            nsample=64,
            npoint=args.preenc_npoints,
            mlp=mlp_dims,
            normalize_xyz=True,
        )
    return preencoder


def build_encoder(args):
    if "ViT-L" in args.clip_vit:
        args.enc_dim = 1024
    encoder = CLIPVITEncoder(vit_model=args.clip_vit, embed_dim=args.enc_dim)
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder


def build_3detr(args, dataset_config):
    print(args)
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        args.dataset_name,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
        use_task_emb=args.use_task_emb,
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor


def build_epcl_encoder(
    pretrain=True, store_path="../model_zoo/EPCL_ckpts/", num_token=256, device="cpu"
):
    if pretrain and not os.path.exists(store_path):
        raise ValueError(f"EPCL pretrained model not found at [{store_path}]!")
    ckpt = torch.load(store_path, map_location=device)
    args = ckpt["args"]
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    model = Model3DETR_encoder(
        pre_encoder,
        encoder,
        dataset_name=args.dataset_name,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        num_queries=args.nqueries,
        use_task_emb=args.use_task_emb,
        vit_only=True,
    )
    print("===> Loading\n", model.__str__)
    match_keys = []
    for key in model.state_dict().keys():
        if key in ckpt["model"]:
            match_keys.append(key)
    model.load_state_dict(ckpt["model"], strict=False)
    print(
        f"===> Loaded\n {store_path}; {len(match_keys)} keys matched; {len(model.state_dict().keys())} keys in total."
    )

    return model.to(device)

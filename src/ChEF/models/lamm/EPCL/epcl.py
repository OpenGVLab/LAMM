import torch.nn as nn
import torch
from ..CLIP import clip as CLIP


def get_frozen_vit(vit_model="ViT-B/32", device="cpu"):
    clip_vit = CLIP.load(vit_model, device=device)[0].visual
    for _, v in clip_vit.named_parameters():
        v.requires_grad = False
    return clip_vit


class CLIPVITEncoder(nn.Module):
    def __init__(self, vit_model="ViT-B/32", embed_dim=768):
        super().__init__()

        """
        vit-b/32 configs: d_model=768, heads=768//64=12, layer=12
        vit-l/14 configs: d_model=1024. heads=1024//64=16, 
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim)
        )
        # clip vit
        vit_drop = 0.3
        self.drop_out = nn.Dropout(vit_drop)
        print(f"[!] Loading CLIP-{vit_model} for EPCL...")
        self.clip_vit = get_frozen_vit(vit_model=vit_model, device=device)
        self.clip_vit.float()
        self.clip_vit.eval()

    def forward(self, x, xyz, task_emb=None):
        B, N, C = x.shape
        pos = self.pos_embed(xyz)

        self.clip_vit.eval()
        x = self.drop_out(x + pos)
        x = self.clip_vit.ln_pre(x)

        if task_emb is not None:
            x = torch.cat([x, task_emb], dim=1)
        x = x.permute(1, 0, 2)
        x = self.clip_vit.transformer(x)
        x = x.permute(1, 0, 2)[:, :N, :].contiguous()

        return xyz, x, None


class TaskEmbEncoder(torch.nn.Module):
    def __init__(self, token_num=40, emb_dim=384):
        super().__init__()
        # Use a two-layer MLP to encode the prefix
        self.embedding = torch.nn.Embedding(token_num, emb_dim)
        # self.embedding = torch.nn.Linear(1,hidden_size,bias=False)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            # torch.nn.Tanh(),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, te: torch.Tensor):
        te_tokens = self.embedding(te)
        past_key_values = self.trans(te_tokens)

        return past_key_values

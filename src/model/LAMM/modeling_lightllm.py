# This script is based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

""" LightLLM LLaMA model, compatible with hf"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import distributed as dist

from safetensors import safe_open
from peft import LoraConfig, TaskType
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.modeling_utils import (PreTrainedModel,
                                         GenerationMixin)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.generation import GenerationConfig

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama2.model import Llama2TpPartModel

from lightllm.common.basemodel.layer_weights.transformer_layer_weight import TransformerLayerWeight

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LightLLMLlamaConfig"

def transformer_layer_load_qkvo(transformerLayerWeight: TransformerLayerWeight, lora_weights, lora_config: LoraConfig):
    lora_scaling_ = lora_config.lora_alpha / lora_config.r
        
    n_embed = transformerLayerWeight.network_config_["hidden_size"]
    split_n_embed = n_embed // transformerLayerWeight.world_size_
    q_lora_A_weight_ = lora_weights[f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.q_proj.lora_A.default.weight']
    q_lora_B_weight_ = lora_weights[f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.q_proj.lora_B.default.weight']
    q_lora_weight_ = torch.mm(q_lora_B_weight_.cuda(), q_lora_A_weight_.cuda()) * lora_scaling_
    q_lora_weight_ = q_lora_weight_[split_n_embed * transformerLayerWeight.tp_rank_: split_n_embed * (transformerLayerWeight.tp_rank_ + 1), :]
    q_lora_weight_ = q_lora_weight_.transpose(0, 1).contiguous().to(transformerLayerWeight.data_type_)
    transformerLayerWeight.q_weight_ += q_lora_weight_

    k_lora_A_weight_ = lora_weights[f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.k_proj.lora_A.default.weight']
    k_lora_B_weight_ = lora_weights[f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.k_proj.lora_B.default.weight']
    k_lora_weight_ = torch.mm(k_lora_B_weight_.cuda(), k_lora_A_weight_.cuda()) * lora_scaling_
    k_lora_weight_ = k_lora_weight_[split_n_embed * transformerLayerWeight.tp_rank_: split_n_embed * (transformerLayerWeight.tp_rank_ + 1), :]
    k_lora_weight_ = k_lora_weight_.transpose(0, 1).contiguous().to(transformerLayerWeight.data_type_)
    transformerLayerWeight.k_weight_ += k_lora_weight_

    v_lora_A_weight_ = lora_weights[f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.v_proj.lora_A.default.weight']
    v_lora_B_weight_ = lora_weights[f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.v_proj.lora_B.default.weight']
    v_lora_weight_ = torch.mm(v_lora_B_weight_.cuda(), v_lora_A_weight_.cuda()) * lora_scaling_
    v_lora_weight_ = v_lora_weight_[split_n_embed * transformerLayerWeight.tp_rank_: split_n_embed * (transformerLayerWeight.tp_rank_ + 1), :]
    v_lora_weight_ = v_lora_weight_.transpose(0, 1).contiguous().to(transformerLayerWeight.data_type_)
    transformerLayerWeight.v_weight_ += v_lora_weight_

    o_lora_A_weight_ = lora_weights[f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.o_proj.lora_A.default.weight']
    o_lora_B_weight_ = lora_weights[f'llama_model.base_model.model.model.layers.{transformerLayerWeight.layer_num_}.self_attn.o_proj.lora_B.default.weight']
    o_lora_weight_ = torch.mm(o_lora_B_weight_.cuda(), o_lora_A_weight_.cuda()) * lora_scaling_
    o_lora_weight_ = o_lora_weight_[split_n_embed * transformerLayerWeight.tp_rank_: split_n_embed * (transformerLayerWeight.tp_rank_ + 1), :]
    o_lora_weight_ = o_lora_weight_.transpose(0, 1).contiguous().to(transformerLayerWeight.data_type_)
    transformerLayerWeight.o_weight_ += o_lora_weight_

def merge_lora_weights(
        lora_weight_path, 
        lora_config: LoraConfig, 
        pre_post_layer=None, 
        transformer_layer_list: List[TransformerLayerWeight]=None
    ):
    use_safetensors = lora_weight_path.endswith('.safetensors')
    if use_safetensors:
        lora_weights = safe_open(lora_weight_path, 'pt', 'cpu')
        lora_weights = {k: lora_weights.get_tensor(k) for k in lora_weights.keys()}
    else:
        lora_weights = torch.load(lora_weight_path, 'cpu')

        if pre_post_layer is not None:
            # TODO
            pass
        if transformer_layer_list is not None:
            # TODO
            for layer in transformer_layer_list:
                transformer_layer_load_qkvo(layer, lora_weights, lora_config)

    return



class LlamaModel:

    def __init__(self,
                 batch_size,
                 max_input_len,
                 max_output_len,
                 weight_dir,
                 lora_path=None,
                 lora_config: LoraConfig=None):
        super().__init__()
        if 'llama2' in weight_dir:
            model_cls = Llama2TpPartModel
        else:
            model_cls = LlamaTpPartModel
        
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()

        logger.info("Initializing ligtllm model.")
        self.base_model: TpPartBaseModel = model_cls(
            tp_rank = self.local_rank, 
            world_size = self.world_size, 
            max_total_token_num= batch_size * (max_input_len + max_output_len), 
            weight_dir=weight_dir, 
            load_way="HF",
        )

        if lora_path is not None:
            merge_lora_weights(lora_path,
                               lora_config,
                               pre_post_layer=None,
                               transformer_layer_list=self.base_model.trans_layers_weight)
        
        self.dtype = torch.float16


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def embed_tokens(self, input_ids):
        embed_tokens = self.base_model.pre_infer.token_forward(
            input_ids = input_ids,
            infer_state = None, 
            layer_weight = self.base_model.pre_post_weight)
        return embed_tokens
    
    def forward(self, *args, **kwds) -> torch.Tensor:

        logits = self.base_model.forward(*args, **kwds)
        return logits


class LlamaLightForCausalLM(GenerationMixin):
    
    main_input_name = "input_ids"

    def __init__(self,
                 batch_size,
                 max_input_len,
                 max_output_len,
                 weight_dir,
                 lora_path=None,
                 lora_config: LoraConfig=None):
        super().__init__()

        self.model = LlamaModel(
                 batch_size,
                 max_input_len,
                 max_output_len,
                 weight_dir,
                 lora_path=lora_path,
                 lora_config=lora_config)
        self.infer_state: Dict = None

        
        with open(os.path.join(weight_dir, 'config.json'), 'r') as f:
            config_json = json.load(f)
        self.config = LlamaConfig(**config_json)
        with open(os.path.join(weight_dir, 'generation_config.json'), 'r') as f:
            generation_config_json = json.load(f)
        self.generation_config = GenerationConfig(**generation_config_json)
        
        self.device = torch.device('cuda')
        self.dtype = torch.float16


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        raise NotImplemented

    def set_output_embeddings(self, new_embeddings):
        raise NotImplemented

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
        if "GenerationMixin" in str(self.prepare_inputs_for_generation.__func__):
            return False
        return True
    
    def init_infer_state(self,
        batch_size,
        total_token_num,
        max_input_len,
    ):
        self.infer_state = dict(
            is_prefill = True,
            batch_size = batch_size,
            total_token_num = total_token_num,
            max_len_in_batch = max_input_len,
        )
    def update_infer_state(self):
        self.infer_state['total_token_num'] += self.infer_state['batch_size']
        self.infer_state['max_len_in_batch'] += 1
        self.infer_state['is_prefill'] = False
    
    def reset_infer_state(self,
    ):
        self.infer_state = None

    def init_buffer(self,
        per_input_len,
        max_input_len,
        max_output_len,
    ):
        batch_size = self.infer_state['batch_size']
        self.b_loc = torch.zeros(batch_size, max_input_len + max_output_len, dtype=torch.long, device="cuda")
        self.b_seq_len = torch.as_tensor(per_input_len, dtype=torch.int32, device="cuda")
        self.b_start_loc = torch.cumsum(torch.as_tensor([0] + per_input_len[:-1], dtype=torch.int32, device="cuda"), dim=0)
        return
    
    def update_buffer(self,

    ):
        batch_size = self.infer_state['batch_size']
        self.b_seq_len += 1
        self.b_start_loc += torch.arange(0, batch_size, dtype=torch.int32, device=self.b_start_loc.device)
        return
    
    def empty_buffer(self):
        assert self.infer_state is not None
        batch_size = self.infer_state['batch_size']
        max_input_len = self.infer_state['max_len_in_batch']
        for i in range(batch_size):
            self.model.base_model.mem_manager.free(
                self.b_loc[i, max_input_len - self.b_seq_len[i]:max_input_len]
            )
        return

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        query_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        outputs = self.model.base_model.forward(
            input_ids=input_ids,
            b_loc=self.b_loc,
            b_start_loc=self.b_start_loc,
            b_seq_len=self.b_seq_len,
            input_embs=inputs_embeds,
            **self.infer_state,
        )

        # hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states)
        logits = outputs.unsqueeze(1) # [B 1 DN]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        query_embeds=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                query_embeds = None

        per_input_len :torch.Tensor = attention_mask.sum(1).tolist()
        batch_size = input_ids.shape[0]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and self.infer_state is None:
            C = inputs_embeds.shape[-1]
            attention_mask = attention_mask.view(batch_size, -1, 1).to(dtype=bool)
            inputs_embeds = torch.masked_select(inputs_embeds, attention_mask)
            model_inputs = {
                "inputs_embeds": inputs_embeds.view(-1, C)
            }

        if self.infer_state == None:
            self.init_infer_state(
                batch_size=batch_size,
                total_token_num = sum(per_input_len),
                max_input_len = max(per_input_len),
            )
            self.init_buffer(
                per_input_len=per_input_len,
                max_input_len=max(per_input_len),
                max_output_len=400,
            )
        else:
            self.update_infer_state()
            self.update_buffer()
            model_inputs = {
                "input_ids": input_ids[:, -1]
            }

        model_inputs.update(
            {
                "position_ids": position_ids,
                "query_embeds": query_embeds,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # override
    def sample(self, *args, **kwds):
        ret =  super().sample(*args, **kwds)
        self.empty_buffer()
        self.reset_infer_state()
        return ret


    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past

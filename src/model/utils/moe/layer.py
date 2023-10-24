import math
from peft.tuners.lora import LoraLayer
from peft.utils import transpose
import torch
import torch.nn as nn
import torch.nn.functional as F


class Top2Gating(nn.Module):
    MIN_EXPERT_CAPACITY = 4
    
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
    ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    @staticmethod
    def top1(tensor):
        values, index = tensor.topk(k=1, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        return values, index

    def forward(self, x, reduce_token=False):
        *_, b, group_size, dim = x.shape
        if reduce_token:
            group_size = 1
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        if reduce_token:
            raw_gates = raw_gates.mean(dim=1).unsqueeze(dim=1)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = self.top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        
        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = self.top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        if reduce_token:
            soft_gate = torch.zeros(b, num_gates).to(gate_1.device)
            soft_gate = soft_gate.to(gate_1.dtype)
            soft_gate.scatter_(1, index_1, gate_1)
            soft_gate = soft_gate.to(gate_2.dtype)
            soft_gate.scatter_(1, index_2, gate_2)
        
        else:
            soft_gate = torch.zeros(b * group_size, num_gates).to(gate_1.device)
            soft_gate = soft_gate.to(gate_1.dtype)
            soft_gate.scatter_(1, index_1.view(-1, 1), gate_1.view(-1, 1))
            soft_gate = soft_gate.to(gate_2.dtype) 
            soft_gate.scatter_(1, index_2.view(-1, 1), gate_2.view(-1, 1))
            soft_gate = soft_gate.reshape(b, group_size, num_gates).contiguous()

        return soft_gate
    

class MoeLoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.lora_moe_A = nn.ParameterDict()
        self.lora_moe_B = nn.ParameterDict()
        
    def update_moe_layer(
        self, 
        adapter_name, 
        r, 
        num_experts, 
        lora_alpha, 
        lora_dropout, 
        init_lora_weights
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        if r > 0:
            lora_A = nn.Parameter(torch.zeros(num_experts, self.in_features, r))
            lora_B = nn.Parameter(torch.zeros(num_experts, r, self.out_features))
            self.lora_moe_A.update(nn.ParameterDict({adapter_name: lora_A}))
            self.lora_moe_B.update(nn.ParameterDict({adapter_name: lora_B}))
            
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_moe_top2_layer(
        self, 
        adapter_name, 
        r, 
        num_experts, 
        lora_alpha, 
        lora_dropout, 
        init_lora_weights
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        if r > 0:
            lora_A = nn.Parameter(torch.zeros(num_experts, self.in_features, r))
            lora_B = nn.Parameter(torch.zeros(num_experts, r, self.out_features))
            self.lora_moe_A.update(nn.ParameterDict({adapter_name: lora_A}))
            self.lora_moe_B.update(nn.ParameterDict({adapter_name: lora_B}))
            
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_moe_A.keys():
            nn.init.kaiming_uniform_(self.lora_moe_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_moe_B[adapter_name])

        if adapter_name in self.lora_moe_A.keys():
            nn.init.kaiming_uniform_(self.lora_moe_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_moe_B[adapter_name])


class MoeLinear(nn.Linear, MoeLoraLayer):

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        num_experts: int = 16,
        gate_mode: str = 'top2_gate',
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MoeLoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        
        nn.Linear.reset_parameters(self)
        self.active_adapter = adapter_name

        self.num_experts = num_experts
        self.gate_mode = gate_mode
        self.moe_gate = None
        assert self.gate_mode in ['top2_gate']
        self.update_moe_top2_layer(
                adapter_name, r, num_experts, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self):
        raise NotImplementedError
    
    def unmerge(self):
        raise NotImplementedError
    
    def set_gate(self, gate: torch.tensor):
        self.moe_gate = gate
    
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys() and self.active_adapter not in self.lora_moe_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_moe_A[self.active_adapter].dtype)
            out = self.lora_dropout[self.active_adapter](x)
            out = torch.einsum('bnd,edh->bneh', out, self.lora_moe_A[self.active_adapter])
            out = torch.einsum('bneh,ehd->bned', out, self.lora_moe_B[self.active_adapter])

            if self.gate_mode == 'top2_gate':  # global sample-level gate
                assert self.moe_gate is not None
                soft_gate = self.moe_gate
                out = torch.einsum('bned,be->bned', out, soft_gate)
            
            out = out.sum(dim=2)
            result += out

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result

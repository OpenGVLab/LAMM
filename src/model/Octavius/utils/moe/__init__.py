import enum
import peft
from peft import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING


# register MoE LoRA
class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    MOE_LORA = 'MOE_LORA'

peft.PeftType = PeftType

from .moe_lora import MoeLoraConfig, MoeLoraModel
PEFT_TYPE_TO_CONFIG_MAPPING[peft.PeftType.MOE_LORA] = MoeLoraConfig
PEFT_TYPE_TO_MODEL_MAPPING[peft.PeftType.MOE_LORA] = MoeLoraModel


__all__ = [
    'MoeLoraConfig', 
]

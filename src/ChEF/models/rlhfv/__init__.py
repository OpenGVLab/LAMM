import torch
from .llava import LlavaLlamaForCausalLM
from .muffin import Beit3LlavaLlamaForCausalLM
from .conversation import conv_templates
from transformers import AutoTokenizer, AutoConfig, StoppingCriteria
from .utils import build_transform
import os

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


def expand_question_into_multimodal(question_text, image_token_len, im_st_token, im_ed_token, im_patch_token):
    if '<image>' in question_text:
        question_text = question_text.replace('<image>', '')

    question_text = question_text + '\n' + im_st_token + im_patch_token * image_token_len + im_ed_token
    return question_text


def wrap_question_with_default_conv(question_text, image_token_len):
    question_text = expand_question_into_multimodal(
        question_text, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)
    conv = conv_templates['default'].copy()
    conv.messages = []
    conv.sep = '\n###'

    conv.append_message(conv.roles[0], question_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def torch_pad_sequence(sequence, padding_value, batch_first=True, padding_side='right'):

    if padding_side == 'right':
        sequence = torch.nn.utils.rnn.pad_sequence(
            sequence,
            batch_first=batch_first,
            padding_value=padding_value)
    elif padding_side == 'left':
        sequence = torch.nn.utils.rnn.pad_sequence(
            [v.flip(-1) for v in sequence],
            batch_first=batch_first,
            padding_value=padding_value)
        sequence = sequence.flip(-1)
    else:
        raise NotImplementedError(f'padding_size={padding_side}')
    return sequence


def qa_colloator_fn(data_list, tokenizer, img_transform):
    questions = [x['question'] for x in data_list]
    tokenized = tokenizer(questions)

    input_ids = [torch.as_tensor(v) for v in tokenized['input_ids']]
    input_ids = torch_pad_sequence(input_ids, tokenizer.pad_token_id, padding_side='left')

    attn_mask = [torch.as_tensor(v) for v in tokenized['attention_mask']]
    attn_mask = torch_pad_sequence(attn_mask, 0, padding_side='left')

    images = [img_transform(x['image']) for x in data_list]
    images = torch.stack(images)

    raw_questions = [x['raw_question'] for x in data_list]
    data = {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attn_mask,
        'raw_questions': raw_questions
    }

    if 'question_id' in data_list[0]:
        data['question_id'] = [x['question_id'] for x in data_list]

    return data


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_size):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.input_size = input_size

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for o in output_ids:
            o = self.tokenizer.decode(o[self.input_size:], skip_special_tokens=True)
            if all([keyword not in o for keyword in self.keywords]):
                return False
        return True

def init_muffin(model_path, device = None):
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Load muffin model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    patch_config(model_name)
    model = Beit3LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map={"":device})
    image_processor = build_transform(
        is_train=False, input_size=model.model.vision_tower.args.img_size)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.model.vision_tower
    if device is not None:
        vision_tower.to(device=device, dtype=torch.float16)
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)

    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = model.model.config.num_query

    return model, image_processor, image_token_len, tokenizer
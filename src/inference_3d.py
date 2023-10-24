import os
from model import LAMMPEFTModel, Octavius
import torch
import json
import argparse
from conversations import conv_templates
from tqdm import tqdm
from bigmodelvis import Visualization
from datasets import load_3Deval_dataset, load_3Deval_dataset_v2
from datasets.system_msg import common_task2sysmsg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openllama_peft')
    parser.add_argument(
        "--encoder_pretrain",
        type=str,
        default="epcl",
        choices=("clip", "epcl"),
        help="Vision Pretrain Model",
    )
    parser.add_argument(
        "--encoder_ckpt_path",
        type=str,
        help="path of vision pretrained model; CLIP use default path in cache",
    )
    parser.add_argument(
        "--llm_ckpt_path",
        type=str,
        required=True,
        help="path of LLM, default: Vicuna",
    )
    parser.add_argument(
        "--delta_ckpt_path",
        type=str,
        help="path of delta parameters from previous stage; Only matter for stage 2",
    )
    parser.add_argument('--stage', type=int, default=2,)
    # Octavius MoE configurations
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--moe_lora_num_experts', type=int, default=4)
    parser.add_argument('--moe_gate_mode', type=str, default='top2_gate')
    parser.add_argument('--octavius_modality', nargs='+', default=['image', 'pcl'])
    # Point Cloud Modality configuration
    parser.add_argument('--num_query_rsp_3d', type=int, default=16)
    parser.add_argument('--hidden_size_rsp_3d', type=int, default=768)
    parser.add_argument('--num_layers_rsp_3d', type=int, default=1)
    parser.add_argument('--num_heads_rsp_3d', type=int, default=8)
    # LoRA configurations
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    # Embedding configurations
    parser.add_argument('--vision_feature_type', type=str, default='local', choices=('local', 'global'))
    parser.add_argument('--vision_output_layer', type=int, default=-1, choices=(-1, -2), help='the layer to output visual features; -1 means global from last layer')
    parser.add_argument('--num_vision_token', type=int, default=1) # the maximum sequence length
    # Test configurations
    parser.add_argument('--max_tgt_len', type=int, default=400, help="maximum length of target sequence at least 400; in case of 1 vision token")
    parser.add_argument('--conv_mode', type=str, default='simple')
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task-name", type=str, default='VQA')  # choose one from [VQA, Caption, VG, Classification]
    parser.add_argument("--base-data-path", required=True)
    parser.add_argument("--vision-root-path", required=True)
    parser.add_argument("--inference-mode", default='common')
    parser.add_argument("--bs", type=int,default=1)
    parser.add_argument("--answers-dir", required=True)
    args = parser.parse_args()

    if args.vision_feature_type == 'local':
        args.vision_output_layer = -2
        args.num_vision_token = 256
    elif args.vision_feature_type == 'global':
        args.vision_output_layer = -1
        args.num_vision_token = 1
    else:
        raise NotImplementedError('NOT implement vision feature type: {}'.format(args.vision_feature_type))
    
    assert os.path.exists(args.delta_ckpt_path), "delta checkpoint not exists!"
    assert os.path.exists(args.llm_ckpt_path), "vicuna checkpoint not exists!"
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    return args


single_infernce_dataset = [
    'ScanNet', 'ScanRefer', 'ScanQA_multiplechoice'
]


def generate_conversation_text(args, input_list, history, sys_msg=None):
    """get all conversation text

    :param args args: input args
    :param str question: current input from user
    :param list history: history of conversation, [(q, a)]
    """
    conv = conv_templates[args.conv_mode]
    if sys_msg:
        conv.system = sys_msg
    prompts_list = []
    for input in input_list:
        prompts = ''
        prompts += conv.system 
        for q, a in history:
            prompts += "{} {}: {}\n{} {}: {}\n".format(conv.sep, conv.roles[0], q, conv.sep, conv.roles[1], a)
        prompts += "{} {}: {}\n".format(conv.sep, conv.roles[0], input)
        prompts_list.append(prompts)
    return prompts_list


def predict(
    args,
    model,
    input,
    history, 
    sys_msg,
):
    prompt_text = generate_conversation_text(args, [input["output_texts"][0][0]['value']], history, sys_msg)
    input['prompt'] = prompt_text
    response = model.generate(input)
    history.append((input, response))
    return history


def default_response(args,
                    model,
                    input,
                    sys_msg):
    """get response text by default

    :param args args: input arguments
    :param model model: model class
    :param input input: input text
    :param object pcl_paths: image objects
    :param str sys_msg: system message for test
    :return list: list of response
    """
    update_param = {
        'top_p': 0.9,
        'temperature': 1.0,
        'max_tgt_len': args.max_tgt_len,
        'modality_embeds': []
    }
    input.update(update_param)
    history = predict(
        args=args,
        model=model,
        input=input,
        history=[],
        sys_msg=sys_msg,
    )
    response = history[-1][1]
    ans_list = []
    
    for res in response:
        ans_list.append(res.split('###')[0])
    return ans_list


def build_model(args):
    model_name = args.model
    if model_name == 'openllama_peft':
        model = LAMMPEFTModel(**args.__dict__)
    elif model_name == 'octavius':
        model = Octavius(**args.__dict__)
    else:
        raise ValueError(f'model name {model_name} not found.')
    
    return model


def main(args):
    # load model
    model = build_model(args)
    delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    if not args.peft_type == 'moe_lora':
        print(f'[!] merging LoRA weights ...')
        model.llama_model = model.llama_model.merge_and_unload()
    model = model.eval().half().cuda()
    Visualization(model).structure_graph()
    print(f'[!] init the LLM over ...')
    
    # load data
    dataset_name = args.dataset_name
    batch_size = args.bs
    if dataset_name in single_infernce_dataset:
        batch_size = 1
    dataloader = load_3Deval_dataset_v2(
        args.base_data_path,
        args.task_name,
        args.dataset_name,
        args.vision_root_path,
        batch_size = batch_size
    )

    answers_file_name = args.task_name + '_' + args.dataset_name + '.json'
    answers_file = os.path.join(args.answers_dir, answers_file_name)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_list = []
    ans_file = open(os.path.splitext(answers_file)[0] + '.jsonl', 'w')
    for data_item in tqdm(dataloader):
        system_msg = common_task2sysmsg[args.task_name + '3D']
        ans_dict = {"scene_id": data_item['scene_id'][0],
            "question": data_item["output_texts"][0][0]['value'],
        }
        response = default_response(args, model, data_item, system_msg)
        ans_dict['text'] = response
        ans_list.append(ans_dict)
        ans_file.write(json.dumps(ans_dict) + "\n")
        ans_file.flush()

    ans_file.close()
    # dump all
    ans_file = open(answers_file, "w")
    ans_file.write(json.dumps(ans_list, indent=4))
    ans_file.flush()
    ans_file.close()
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

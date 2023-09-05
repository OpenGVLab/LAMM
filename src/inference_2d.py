import os
from model.openlamm import LAMMPEFTModel
import torch
from torch import distributed as dist
import json
import argparse
from conversations import conv_templates
from tqdm import tqdm
from bigmodelvis import Visualization
from datasets import load_2Deval_dataset

answers_file = ''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openllama_peft')
    parser.add_argument(
        "--encoder_pretrain",
        type=str,
        default="clip",
        choices=("clip", "epcl"),
        help="Vision Pretrain Model",
    )
    parser.add_argument(
        "--encoder_ckpt_path",
        type=str,
        default="",
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
    parser.add_argument("--base-data-path", required=True)
    parser.add_argument("--inference-mode", default='common')
    parser.add_argument("--bs", type=int,default=1)
    parser.add_argument("--answers-dir", required=True)
    parser.add_argument("--use-lightllm", default=False, action="store_true", help="whether to use lightllm in inference to speed up",)
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
    assert len(args.encoder_ckpt_path) == 0 or os.path.exists(args.encoder_ckpt_path), "vision encoder checkpoint not exists!"
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    return args


# single_infernce_dataset = [
#     'FSC147','SQAimage','AI2D'
# ]
keypoints_definition = ["right ankle", "right knee", "right hip", "left hip", "left knee" ,"left ankle" ,"right wrist", "right elbow", "right shoulder", "left shoulder" ,"left elbow", "left wrist", "neck", "head top" ]


def generate_conversation_text(args, input_list, history, sys_msg = None):
    """get all conversation text

    :param args args: input args
    :param str question: current input from user
    :param list history: history of conversation, [(q, a)]
    """
    conv = conv_templates[args.conv_mode]
    if sys_msg:
        if conv.sys_temp is not None:
            conv.system = conv.sys_temp.format(system_message=sys_msg)
        else:
            conv.system = sys_msg
    prompts_list = []
    for input in input_list:
        prompts = ''
        prompts += conv.system + '\n\n'
        for q, a in history:
            prompts += "{}: {}\n{} {}: {}\n{}".format(conv.roles[0], q, conv.sep, conv.roles[1], a, conv.sep2 if (conv.sep2 is not None) else conv.sep)
        prompts += "{}: {}\n{}".format(conv.roles[0], input, conv.sep)
        prompts_list.append(prompts)
    return prompts_list


def predict(
    args,
    model,
    input,
    images, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    sys_msg,
):
    prompt_text = generate_conversation_text(args, input, history, sys_msg)
    response = model.generate({
        'prompt': prompt_text,
        'images': images,
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': []
    })
    history.append((input, response))
    return history


def default_response(args,
                    model,
                    input,
                    images,
                    sys_msg):
    history = predict(
        args=args,
        model=model,
        input=input,
        images=images,
        max_length=args.max_tgt_len,
        top_p=0.9,
        temperature=1.0,
        history=[],
        sys_msg = sys_msg,
    )
    conv = conv_templates[args.conv_mode]
    response = history[-1][1]
    ans_list = []
    for res in response:
        ans_list.append(res.split(conv.sep2 if conv.sep is not None else conv.sep)[0])
    return ans_list


def vqa_response(args,
                model,
                input,
                images,
                sys_msg):
    conv = conv_templates[args.conv_mode]
    reasoning_list = default_response(args, model, input, images, sys_msg)
    option_prompt = []
    for prompt_1, response_1 in zip(input, reasoning_list):
        option_prompt.append(prompt_1 + response_1 + ' {}\nANSWER:'.format(conv.sep)) 
    final_answer_list = default_response(args, model, option_prompt, images, sys_msg)
    all_answer_list = []
    for reasoning, option in zip(reasoning_list, final_answer_list):
        all_answer_list.append(reasoning + '\n The answer is ' + option)
    return all_answer_list


def kps_det_response(args,
                    model,
                    input,
                    images,
                    sys_msg):
    answer_list = [dict()] * len(input)
    for keypoint_name in keypoints_definition:
        prompts = [text.format(keypoint_name) for text in input]
        keypoint_answer_list = default_response(args, model, prompts, images, sys_msg)
        for i in range(len(input)):
            answer_list[i][keypoint_name] = keypoint_answer_list[i]
    return answer_list


def main(args):
    if args.use_lightllm:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0")) % torch.cuda.device_count()
        dist.init_process_group("nccl", 
                                init_method="tcp://127.0.0.1:10086",
                                world_size=world_size, rank=local_rank)
    # load model
    model = LAMMPEFTModel(**args.__dict__)
    delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    print(f'[!] merging LoRA weights ...')
    if not args.use_lightllm:
        model.llama_model = model.llama_model.merge_and_unload()
    model = model.eval().half().cuda()
    Visualization(model).structure_graph()
    print(f'[!] init the LLM over ...')
    
    # load data
    dataset_name = args.dataset_name
    inference_mode = args.inference_mode
    batch_size = args.bs
    # if dataset_name in single_infernce_dataset:
    #     batch_size = 1
    dataloader = load_2Deval_dataset(
        args.base_data_path,
        args.dataset_name,
        inference_mode,
        batch_size = batch_size
    )
    sys_msg = dataloader.dataset.system_msg
    task_name = dataloader.dataset.task_name

    answers_file_name = task_name + '_' + args.dataset_name + '.json'
    answers_file = os.path.join(args.answers_dir, answers_file_name)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_list = []
    ans_file = open(os.path.splitext(answers_file)[0] + '.jsonl', 'w')
    for data_item in tqdm(dataloader):
        prompt = data_item['query']
        image = data_item['image']
        
        if task_name == 'VQA':
            response_func = vqa_response
        elif task_name == 'Keypoints_Detection':
            response_func = kps_det_response
        else:
            response_func = default_response
        
        answer_list = response_func(
            args=args,
            model=model,
            input=prompt,
            images=image,
            sys_msg=sys_msg,
        )

        for id, output in zip(data_item['id'], answer_list):
            ans_dict = {"id": id,
                        "text": output,
                        "delta_path": args.delta_ckpt_path
                        }
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

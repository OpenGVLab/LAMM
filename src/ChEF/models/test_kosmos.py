import torch
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from collections import namedtuple
import sentencepiece as spm
import ast
from fairseq_cli.generate import get_symbols_to_strip_from_output

import torch.nn.functional as F
from .kosmos2.utils import get_interactive_tokens_and_lengths, post_process_prediction, get_token_src
from .test_base import TestBase
from .kosmos2 import unilm

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints img_src_tokens img_gpt_input_mask")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

class TestKOSMOS2(TestBase): # TODO: batch_size = 1
    def __init__(self, model_path, 
                 dict_path = 'ChEF/models/kosmos2/data/dict.txt', 
                 tokenizer_path = 'ChEF/models/kosmos2/data/sentencepiece.bpe.model',
                 if_grounding = True, 
                 device = 'cuda',
                 **kwargs):
        parser = options.get_interactive_generation_parser()
        input_args = ['--local_rank=0', 'None', 
                      '--task', 'generation_obj', 
                      '--path', model_path, 
                      '--dict-path', dict_path, 
                      '--required-batch-size-multiple', '1', 
                      '--remove-bpe=sentencepiece', 
                      '--max-len-b', '500', 
                      '--add-bos-token', 
                      '--beam', '1', 
                      '--buffer-size', '1', 
                      '--image-feature-length', '64', 
                      '--locate-special-token', '1', 
                      '--batch-size', '1', 
                      '--nbest', '1', 
                      '--no-repeat-ngram-size', '3', 
                      '--location-bin-size', '32']
        
        # buffer_size >= batch_size
        args = options.parse_args_and_arch(parser, input_args=input_args)
        cfg = convert_namespace_to_omegaconf(args)
        cfg['common_eval']['model_overrides'] =  "{'visual_pretrained': '', 'dict_path':'" + dict_path + "'}"
        task = tasks.setup_task(cfg.task)
        self.task = task
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)
        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
        self.model = models[0]
        self.move_to_device(cfg, device)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        self.special_tokens = [self.task.source_dictionary[idx] for idx in range(self.tokenizer.vocab_size(), 
                                    len(self.task.source_dictionary))]
        cfg.generation.sampling = False
        cfg.generation.sampling_topp = -1.0
        cfg.generation.temperature = 1.0
        cfg.generation.beam = 1
        cfg.generation.max_len_a = 1
        self.generator = self.task.build_generator([self.model], cfg.generation, extra_gen_cls_kwargs = dict(ppl = False))
        self.cfg = cfg
        self.if_grounding = if_grounding

    def move_to_device(self, cfg, device):
        self.dtype = torch.float16
        self.device = device
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.prepare_for_inference_(cfg)

    def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None, batch_answers=None, **kwargs):
        prefix = "[image]<image><tab><grounding>" if self.if_grounding else "[image]<image><tab>"
        prompt = " ".join([prefix] * len(image_list)) + prompt
        if CoT_answer_list is not None:
            prompt += ' ' + CoT_answer_list[idx]
        if batch_answers is not None:
            prompt += '\n' + batch_answers[idx]
        return prompt

    def build_input_image(self, image_list):
        images = self.get_image_list(image_list)
        return images

    @torch.no_grad()
    def ppl_inference_multi_imgs(self, image_list, question_list, answer_list, answer_pool, CoT_list = None, calib = False):
        self.generator.ppl = True
        batch_size = len(images)
        images = [get_multi_imgs(image) for image in image_list]
        images = [item for sublist in images for item in sublist]
        self.cfg.dataset.batch_size = batch_size
        prefix = "[image]<image><tab><grounding>" if self.if_grounding else "[image]<image><tab>"
        prompts = [" ".join([prefix] * len(images)) + {question} for question, images in zip(question_list, image_list)]
        for i in range(batch_size):
            answer = ''
            if CoT_list is not None:
                answer = CoT_list[i] + '\n'
            answer += answer_list[i]
            prompts[i] += ' ' + answer
        sample = self.make_batches(images, prompts)
        results = self.do_ppl(sample, answer_list, answer_pool, calib=calib)
        return results

    def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
        self.generator.ppl = False
        self.generator.max_len_b = max_new_tokens
        sample = self.make_batches(image_list, [prompt])
        translations = self.task.inference_step(
            self.generator, [self.model], sample, constraints=None
        )
        results = []
        for i, (id, hypos) in enumerate(zip(sample['ids'].tolist(), translations)):
            src_tokens_i = utils.strip_pad(sample['net_input']['src_tokens'][i], self.task.target_dictionary.pad())
            results.append(
                (
                    id,
                    src_tokens_i,
                    hypos,
                )
            )
        outputs = []
        for id_, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            src_str = self.task.source_dictionary.string(src_tokens, self.cfg.common_eval.post_process)
            # Process top predictions
            for hypo in hypos[: min(len(hypos), self.cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=None,
                    tgt_dict=self.task.target_dictionary,
                    remove_bpe=self.cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )
                hypo_str = hypo_str.replace(src_str, '')
                outputs.append(hypo_str)
        return outputs[0]

    def make_batches(self, images, inputs):
        tokens, lengths, img_src_tokens, img_gpt_input_mask = \
            get_interactive_tokens_and_lengths(self.task, images, inputs, self.tokenizer ,self.special_tokens)
        task = self.task
        cfg = self.cfg
        itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_caption_inference(
                tokens, lengths, img_src_tokens, img_gpt_input_mask
            ),
            max_sentences=cfg.dataset.batch_size,
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        ).next_epoch_itr(shuffle=False)
        res_list = []
        for batch in itr:
            ids = batch["id"]
            src_tokens = batch["net_input"]["src_tokens"].to(self.device)
            src_lengths = batch["net_input"]["src_lengths"].to(self.device)
            img_src_tokens = batch["net_input"]["img_src_tokens"].to(dtype = self.dtype)
            img_gpt_input_mask = batch["net_input"]["img_gpt_input_mask"]
            res_list.append(dict(
                ids = ids,
                net_input = dict(
                    src_tokens = src_tokens,
                    src_lengths = src_lengths,
                    img_src_tokens = img_src_tokens,
                    img_gpt_input_mask = img_gpt_input_mask
                )
            ))
        assert len(res_list) == 1
        return res_list[0]

    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        sample = self.make_batches(batch_images, batch_prompt)
        probs = self.task.inference_step(
            self.generator, [self.model], sample, constraints=None
        )
        logits = probs[:, :-1]
        labels = sample['net_input']['src_tokens'][:, 1:]
        results = []
        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(get_token_src(self.task, option, self.tokenizer, self.special_tokens))
        import ipdb;ipdb.set_trace()
        for idx in range(labels.shape[0]):
            option_len = len(batch_option_ids[idx])
            non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
            start_index = non_zero_indices.max() - option_len + 1
            end_index = start_index + option_len
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == batch_option_ids[idx].numpy()):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
        return results


    def xx(self, sample, answer_list, answer_pool, calib=False):
        answer_start_indices = []
        answer_end_indices = []
        template_token_list = []
        answer_token_list = []
        for template, option in zip(answer_list, answer_pool):
            template_token = get_token_src(self.task, template, self.tokenizer, self.special_tokens)
            template_token_list.append(template_token)
            option_token = get_token_src(self.task, option, self.tokenizer, self.special_tokens)
            token_len = len(option_token)
            for index in range(len(template_token)):
                if template_token[index: index + token_len] == option_token:
                    answer_start_indices.append(index)
                    answer_end_indices.append(index + token_len)
                    answer_token_list.append(option_token)
                    break
            assert len(answer_start_indices) == len(template_token_list), "tokenizer encode answer in template different from answer only"
        probs = self.task.inference_step(
            self.generator, [self.model], sample, constraints=None
        )
        logits = probs[:, :-1]
        target_ids = sample['net_input']['src_tokens'][:, 1:]
        start_indices, end_indices = [], []
        for i in range(len(answer_list)):
            token_len = len(template_token_list[i])
            for index in range(target_ids.shape[1] - token_len, 0, -1):
                if target_ids[i,index: index+token_len].cpu().numpy().tolist() == template_token_list[i]:
                    start_indices.append(index + answer_start_indices[i])
                    end_indices.append(index + answer_end_indices[i])
                    target_ids[i,:index] = -1
                    break
            assert len(start_indices) == (i+1), "tokenizer encode answer different from answer in conversation"
        
        
        results = []
        if calib:
            for idx, item_logits in enumerate(logits):
                score = 0.0
                item_prob = F.softmax(item_logits[start_indices[idx]:end_indices[idx]], dim=-1)
                for jdx in range(end_indices[idx]-start_indices[idx]):
                    score += torch.log(item_prob[jdx, answer_token_list[idx][jdx]]).item()
                score = score/len(answer_token_list[idx])
                results.append(score)
        else:
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), target_ids.reshape(-1),ignore_index=-1, reduction='none')
            loss = loss.reshape(-1, target_ids.shape[1]).float()
            for idx, item_loss in enumerate(loss):
                results.append(item_loss[start_indices[idx]: end_indices[idx]].mean().item())
        return results

    
from torchvision import transforms
import re
import torch
from fairseq import utils
inception_normalize = transforms.Compose(
    [transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
)

def square_transform(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def split_string(string, separators):
    """
    Function to split a given string based on a list of separators.

    Args:
    string (str): The input string to be split.
    separators (list): A list of separators to be used for splitting the string.

    Returns:
    A list containing the split string with separators included.
    """
    pattern = "|".join(re.escape(separator) for separator in separators) 
    result = re.split(f'({pattern})', string)  
    return [elem for elem in result if elem] 


def get_interactive_tokens_and_lengths(self, images, inputs, tokenizer=None, special_tokens = None, incontext_cfg = None):
    """
        tokenizer size : 64000
        dictionary size: 65307
        special_token_size: 1037
    """
    image_feature_length = self.args.image_feature_length # 64
    bos_id = self.dictionary.bos()
    eos_id = self.dictionary.eos()
    boi_id = self.dictionary.index("<image>")
    eoi_id = self.dictionary.index("</image>") 
    
    def convert_one_line(idx, input_str):
        token = []
        img_src_token = []
        img_gpt_input_mask = []
        segments = input_str.split('<tab>')
        token.append(bos_id)
        img_gpt_input_mask.append(0)
        img_id = 0
        for i, segment in enumerate(segments):
            if segment.startswith('[image]'):
                if incontext_cfg is not None and incontext_cfg['use_pic']:
                    image = images[idx * incontext_cfg['ice_num'] + img_id]
                else:
                    image = images[idx]
                image_tensor = square_transform(self.args.input_resolution)(image)
                img_src_token.append(image_tensor)
                token.extend([boi_id] + list(range(4, image_feature_length+4)) + [eoi_id])
                img_gpt_input_mask.extend([0] + [1] * image_feature_length + [0])
                img_id += 1
            else:
                split_special_token_words = []
                split_results = split_string(segment, special_tokens)
                for string in split_results:
                    if string in special_tokens:
                        split_special_token_words.append(string)
                    else:
                        encode_tokens = tokenizer.encode(string, out_type=str)
                        split_special_token_words.extend(encode_tokens)
                segment = ' '.join(split_special_token_words)
                text_tokens = self.source_dictionary.encode_line(
                    segment, add_if_not_exist=False
                ).tolist()
                text_tokens = text_tokens[:-1] # </s> in token
                token.extend(text_tokens)
                img_gpt_input_mask.extend([0] * (len(text_tokens)))
        token.append(eos_id) # add </s>
        assert len(token) == len(img_gpt_input_mask) + 1 
        token = torch.LongTensor(token)
        img_gpt_input_mask = torch.LongTensor(img_gpt_input_mask)
        img_src_token = torch.stack(img_src_token, dim=0)
        return token, img_src_token, img_gpt_input_mask
    
    tokens = []
    img_src_tokens = []
    img_gpt_input_masks = []
    for idx, src_str in enumerate(inputs):
        token, img_src_token, img_gpt_input_mask = convert_one_line(idx, src_str)
        tokens.append(token)
        img_src_tokens.append(img_src_token)
        img_gpt_input_masks.append(img_gpt_input_mask)
    lengths = [t.numel() for t in tokens]
    
    return tokens, lengths, img_src_tokens, img_gpt_input_masks

def get_token_src(self, input, tokenizer, special_tokens = None):
    split_special_token_words = []
    split_results = split_string(input, special_tokens)
    for string in split_results:
        if string in special_tokens:
            split_special_token_words.append(string)
        else:
            encode_tokens = tokenizer.encode(string, out_type=str)
            split_special_token_words.extend(encode_tokens)
    input = ' '.join(split_special_token_words)
    text_tokens = self.source_dictionary.encode_line(
        input, add_if_not_exist=False
    ).tolist()
    text_tokens = text_tokens[:-1] # </s> in token
    return text_tokens

def remove_special_fields(text):  
    return re.sub('<.*?>', '', text)  
  
def find_phrases(text):  
    phrases = re.finditer('<phrase>(.*?)</phrase>', text)  
    return [(match.group(1), match.start(1), match.end(1)) for match in phrases]  
  
def adjust_phrase_positions(phrases, text):  
    positions = []  
    for phrase, start, end in phrases:  
        adjusted_start = len(remove_special_fields(text[:start]))  
        adjusted_end = len(remove_special_fields(text[:end]))  
        positions.append((phrase, adjusted_start, adjusted_end))  
    return positions  
  
def mark_words(text, phrases):  
    marked_words = []  
  
    words = re.findall(r'\b\w+\b|[.,;?!:()"“”‘’\']', text)  
    word_indices = [match.start() for match in re.finditer(r'\b\w+\b|[.,;?!:()"“”‘’\']', text)]  
  
    for i, word in enumerate(words):  
        if any(start <= word_indices[i] < end for _, start, end in phrases):  
            marked_words.append((word, 'box'))  
        else:  
            marked_words.append((word, None))  
  
    return marked_words  

def merge_adjacent_words(marked_words):  
    merged_words = []  
    current_word, current_flag = marked_words[0]  
  
    for word, flag in marked_words[1:]:  
        if flag == current_flag:  
            current_word += " " + word  
        else:  
            merged_words.append((current_word, current_flag))  
            current_word = word  
            current_flag = flag  
  
    merged_words.append((current_word, current_flag))  
    return merged_words

def mark_texts(text):
    cleaned_text = remove_special_fields(text)      
    phrases = find_phrases(text)  
    adjusted_phrases = adjust_phrase_positions(phrases, text)      
    marked_words = mark_words(cleaned_text, adjusted_phrases)  
    merge_words = merge_adjacent_words(marked_words)
    return merge_words


def post_process_prediction(
    hypo_tokens,
    src_str,
    alignment,
    align_dict,
    tgt_dict,
    remove_bpe=None,
    extra_symbols_to_ignore=None,
):
    hypo_str = tgt_dict.string(
        hypo_tokens, remove_bpe, extra_symbols_to_ignore=extra_symbols_to_ignore
    )
    if align_dict is not None:
        hypo_str = utils.replace_unk(
            hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
        )
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=False)
    return hypo_tokens, hypo_str, alignment
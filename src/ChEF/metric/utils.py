import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
import numpy as np
from nltk.corpus import wordnet as wn
import re
import json


voc_syndict = {
    'man': 'person',
    'men': 'person',
    'woman': 'person',
    'women': 'person',
    'baby': 'person',
    'people': 'person',
    'child': 'person',
    'children': 'person',
    'girl': 'person',
    'boy': 'person',
    'guy': 'person',
    'ship': 'boat',
    'sailboat': 'boat',
    'yacht': 'boat',
    'puppy': 'dog',
    'table': 'diningtable',
    'calf': 'cow',
    'hummingbird': 'bird',
    'bike': 'bicycle',
    'kayaker': 'person',
    'motorcycle': 'motorbike',
    'pony': 'horse'
}

class Base_Metric:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    def metric_func(self, answers):
        pass

    def metric(self, answer_path):
        with open(answer_path, 'rb') as f:
            answers = json.load(f)
        results = self.metric_func(answers) 

        if isinstance(results, dict):
            print(f'{self.dataset_name}:')
            for key, value in results.items():
                print(f'{key}: {value}')
            return results
        else:
            print(f'{self.dataset_name}: {results}')
            return results

class Cleaner:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
    
    def clean(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        return answer

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText



num_pattern =  re.compile(r'[0-9]+\.?[0-9]*')

def is_number(s):
    try: 
        float(s)
        return True
    except ValueError: 
        pass 
    return False

def parse_num(num_list, split_char_a, split_char_b, text):
    flag = 0
    tmpnum = ''
    for c in text:
        if c == split_char_a:
            flag = 1
        elif c == split_char_b:
            flag = 0
            if is_number(tmpnum):
                num_list.append(float(tmpnum))
                tmpnum = ''
        elif flag == 0:
            continue
        else:
            if c!= ',' and c!= ' ':
                tmpnum += c
            else:
                if is_number(tmpnum):
                    num_list.append(float(tmpnum))
                    tmpnum = ''
    return num_list

def cal_iou(bbox1, bbox2):
    ixmin = np.maximum(bbox1[0], bbox2[0])
    iymin = np.maximum(bbox1[1], bbox2[1])
    ixmax = np.minimum(bbox1[2], bbox2[2])
    iymax = np.minimum(bbox1[3], bbox2[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih

    uni = ((bbox2[2] - bbox2[0] ) * (bbox2[3] - bbox2[1]) +
            (bbox1[2] - bbox1[0]) *
            (bbox1[3] - bbox1[1]) - inters)
    
    overlaps = inters / uni
    return overlaps

def classification_acc(gt_text, pred_text, syn_judge = True):
    words = pred_text.split(' ')
    if syn_judge:
        syn_set = wn.synsets(gt_text)
        for word in words:
            if word in voc_syndict:
                words.append(voc_syndict[word])
        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(word, 'n') for word in words]
        syn_list = []
        for syn in syn_set:
            syn_list += syn.lemma_names()
        for syn in syn_list:
            if syn in words:
                return True
    else:
        if gt_text in words:
            return True
    return False

def parse_bbox(text):
    num_list = []
    num_list = parse_num(num_list, '[', ']', text)
    num_list = parse_num(num_list, '(', ')', text)
    bbox_list = []
    num_list = num_list[:(len(num_list)//4) *4]
    if len(num_list) == 0:
        str_list = num_pattern.findall(text)
        num_list = [float(item) for item in str_list]
        num_list = num_list[:(len(num_list)//4) *4]
    for i in range(0,len(num_list), 4):
        cur_bbox = [num_list[j] for j in range(i,i+4)]
        if cur_bbox[0] > cur_bbox[2] and cur_bbox[1] > cur_bbox[3]:
            cur_bbox = [cur_bbox[2],cur_bbox[3], cur_bbox[0], cur_bbox[1]]
        if cur_bbox[0] <= cur_bbox[2] and cur_bbox[1] <= cur_bbox[3]:
            bbox_list.append(cur_bbox)
    return bbox_list


def parse_kosmos_bbox(text):
    def extract_strings_between_tags(string):
        pattern = r"<object>(.*?)</object>"
        matches = re.findall(pattern, string)
        return matches
    def extract_numbers_from_patch(string):
        pattern = r"<patch_index_(\d+)>"
        matches = re.findall(pattern, string)
        return matches
    def index_to_normalized_coordinate(index):
        row = index // 32  # 计算行号
        col = index % 32  # 计算列号
        normalized_y = row / 32  # 归一化行号
        normalized_x = col / 32  # 归一化列号
        return normalized_x, normalized_y
    matches = extract_strings_between_tags(text)
    num_list = []
    for match in matches:
        index_list = extract_numbers_from_patch(match)
        index_list = index_list[:(len(index_list)//2) *2]
        index_list = [int(index) for index in index_list]
        for index in index_list:
            x, y = index_to_normalized_coordinate(index)
            num_list += [x, y]
    num_list = num_list[:(len(num_list)//4) *4]
    bbox_list = []
    for i in range(0,len(num_list), 4):
        cur_bbox = [num_list[j] for j in range(i,i+4)]
        bbox_list.append(cur_bbox)
    return bbox_list
    

def parse_caption_sentence(text):
    pattern = r"(?<=['\"])(.*?)(?=['\"])"
    sentences = re.findall(pattern, text)
    if len(sentences) > 0:
        return '. '.join(sentences)
    return text

num_dict = {
    'no': 0, 'zero':0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine' : 9, 'ten':10, 'eleven':11 , 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
}

def ennum2numerical(text):
    text = text.lower()
    for word in text.split():
        if word.isdigit():
            return int(word)
        if word in num_dict:
            return num_dict[word]
    return 0

if __name__ == "__main__":
    test = '<object><patch_index_0035><patch_index_1023></object> at the station <object><patch_index_0035><patch_index_1023></object> at the station'
    parse_kosmos_bbox(test)

from nltk import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
stops = set(stopwords.words("english"))
import numpy as np
from nltk.corpus import wordnet as wn
import re
num_pattern =  re.compile(r'[0-9]+\.?[0-9]*')


def parse_entity(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in string.punctuation]
    words = [word for word in words if word not in stops]
    words = [wordnet.morphy(word) for word in words if word not in stops]
    return words


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



def cal_iou_3d(bbox1, bbox2):
    '''
        box [x1, y1, z1, l, w, h]
    '''
    bbox1 = [
        round(bbox1[0] - abs(bbox1[3]/2), 3), round(bbox1[1] - abs(bbox1[4]/2), 3), round(bbox1[2] - abs(bbox1[5]/2), 3), 
        round(bbox1[0] + abs(bbox1[3]/2), 3), round(bbox1[1] + abs(bbox1[4]/2), 3), round(bbox1[2] + abs(bbox1[5])/2, 3)
        ]
    
    bbox2 = [
        round(bbox2[0]-abs(bbox2[3]/2),3), round(bbox2[1]-abs(bbox2[4]/2),3), round(bbox2[2]-abs(bbox2[5]/2),3), 
        round(bbox2[0]+abs(bbox2[3]/2),3), round(bbox2[1]+abs(bbox2[4]/2),3), round(bbox2[2]+abs(bbox2[5])/2,3)
        ]
    
    # intersection
    x1, y1, z1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2])
    x2, y2, z2 = min(bbox1[3], bbox2[3]), min(bbox1[4], bbox2[4]), min(bbox1[5], bbox2[5])
    inter_area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    # union
    area1 = (bbox1[3] - bbox1[0]) * (bbox1[4] - bbox1[1]) * (bbox1[5] - bbox1[2])
    area2 = (bbox2[3] - bbox2[0]) * (bbox2[4] - bbox2[1]) * (bbox2[5] - bbox2[2])
    uni_area = area1 + area2 - inter_area
    
    iou = inter_area / uni_area
    
    if iou > 1 or iou < 0:
        return 0
    else:
        return iou
    

def classification_acc(gt_text, pred_text):
    words = parse_entity(pred_text)
    syn_set = wn.synsets(gt_text)
    try:
        syn_list = syn_set[0].lemma_names()
    except:
        syn_list = [gt_text]
    for syn in syn_list:
        if syn in words:
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


def parse_bbox_3d(text):
    num_list = []
    num_list = parse_num(num_list, '[', ']', text)
    num_list = parse_num(num_list, '(', ')', text)
    
    bbox_list = []
    num_list = num_list[:(len(num_list) // 6) * 6]
    if len(num_list) == 0:
        str_list = num_pattern.findall(text)
        num_list = [float(item) for item in str_list]
        num_list = num_list[:(len(num_list)//6) * 6]
    for i in range(0,len(num_list), 6):
        cur_bbox = [num_list[j] for j in range(i, i + 6)]
        
        bbox_list.append(cur_bbox)
    return bbox_list


def parse_keypoints(text):
    num_list = []
    num_list = parse_num(num_list, '[', ']', text)
    num_list = parse_num(num_list, '(', ')', text)
    keypoints_list = []
    num_list = num_list[:(len(num_list)//2) *2]
    if len(num_list) == 0:
        str_list = num_pattern.findall(text)
        num_list = [float(item) for item in str_list]
        num_list = num_list[:(len(num_list)//2) *2]
    for i in range(0,len(num_list), 2):
        cur_kps = [num_list[j] for j in range(i,i+2)]
        keypoints_list.append(cur_kps)
    return keypoints_list


def check_inside_bbox(keypoints_list, bbox):
    for keypoints in keypoints_list:
        x = keypoints[0]
        y = keypoints[1]
        if x>bbox[0] and x < bbox[2] and y>bbox[1] and y < bbox[3]:
            return True
    return False


def point_distance(x, y):
    return np.sqrt(np.sum(np.square(x-y)))


def correct_keypoints(keypoints_list, gt_joint, thr = 0.1):
    for keypoint in keypoints_list:
        if point_distance(np.array(keypoint),gt_joint) < thr:
            return True
    return False


def parse_sentence(text):
    if '\"' in text:
        res = ''
        text_list = text.split('\"')
        for i in range(1,len(text_list), 2):
            res += text_list[i] + ' '
        return res[:-1]
    return text


num_dict = {
    'no': 0, 'zero':0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine' : 9, 'ten':10, 'eleven':11 , 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
}


def ennum2numerical(text):
    for word in text.split():
        if word.isdigit():
            return int(word)
        if word in num_dict:
            return num_dict[word]
    return None
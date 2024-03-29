import os
import json
from tqdm import tqdm
from nltk import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

from .utils import Base_Metric, voc_syndict

stops = set(stopwords.words("english"))


class CG_Classification(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
        from .utils import classification_acc, Cleaner
        self.cleaner = Cleaner()
        self.syn_check_func = classification_acc
    
    def metric_func(self, answers):
        correct = 0
        syn_correct = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            pred = self.cleaner.clean(pred)
            if self.syn_check_func(gt, pred):
                syn_correct += 1
            if gt in pred.split():
                correct += 1
        acc = correct / len(answers) * 100
        syn_acc = syn_correct / len(answers) * 100
        return dict(
            ACC = acc,
            SYN_ACC = syn_acc,
        )

class UCMerced_Classification(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)

    def metric_func(self, answers):
        correct = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            if gt in pred:
                correct += 1
        acc = correct / len(answers) * 100
        return dict(
            ACC = acc,
        )

class LAMM_Classification(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
    
    def metric_func(self, answers):
        def parse_entity(text):
            text = text.lower()
            words = word_tokenize(text)
            words = [word for word in words if word not in string.punctuation]
            words = [word for word in words if word not in stops]
            words = [wordnet.morphy(word) for word in words if word not in stops]
            return words

        def convert_word(input_word):
            conversion_dict = voc_syndict
            lemmatizer = WordNetLemmatizer()
            input_word = lemmatizer.lemmatize(input_word.lower())
            if input_word in conversion_dict:
                return conversion_dict[input_word]
            else:
                return input_word

        def classification_acc_lamm(gt_text, pred_text):
            convert = False
            if convert:
                pred_text = convert_word(pred_text)
            words = parse_entity(pred_text)
            syn_set = wn.synsets(gt_text)
            try:
                syn_list = syn_set[0].lemma_names() + [gt_text]
            except:
                syn_list = [gt_text]
            if pred_text in syn_list:
                return True
            for syn in syn_list:
                if syn in words:
                    return True
            return False
        score = 0.0
        for item in tqdm(answers, desc="Running Metric"):
            gt_label = item['label']
            pred_text = item['answer']
            if classification_acc_lamm(gt_label, pred_text):
                score += 1.0
        return dict(
            ACC = score/len(answers)
        )


class FG_Classification(Base_Metric):
    
    def __init__(self, dataset_name, bamboo_tree_path, inference_type = 'direct', **kwargs):
        super().__init__(dataset_name)         
        self.bamboo_tree_path = bamboo_tree_path
        annot_data = json.load(open(self.bamboo_tree_path,'rb'))
        self.id2name = annot_data['id2name']
        self.father2child = annot_data['father2child']
        name2id = {}
        for key, value in self.id2name.items():
            for name in value:
                name2id[name] = key
        self.name2id = name2id
        self.child2father = annot_data['child2father']
        
        self.inference_type = inference_type
        assert self.inference_type in ['direct', 'single_ppl', 'multi_ppl']

    def weighted_ACC_multi(self, pred, gt):
        assert len(pred) == len(gt)
        for i in range(len(pred)):
            if pred[i] != gt[i]:
                return i/len(pred)
        return 1

    def weighted_ACC_single(self, pred, gt):
        deep_idx = 0
        for i in range(len(gt)):
            if pred == gt[i]:
                deep_idx = i+1
        return deep_idx/len(gt)
    
    def share_father(self, father_id, pred):
        children = self.father2child[father_id]
        for child in children:
            if child in pred:
                return True
        return False

    def weighted_ACC_direct(self, pred, gt):
        deep_idx = 0
        for i in range(len(gt)):
            if gt[i] in pred:
                deep_idx = i+1
            else:
                # whether the class is the brother of the gt
                father = self.name2id[gt[i-1]] if i>0 else self.child2father[self.name2id[gt[i-1]]][0]
                if self.share_father(father, pred):
                    deep_idx = i
        return deep_idx/len(gt)

    def metric_func(self, answers):
        wcorrect = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            if self.inference_type == 'multi_ppl':
                wcorrect += self.weighted_ACC_multi(pred, gt)
            elif self.inference_type == 'single_ppl':
                wcorrect += self.weighted_ACC_single(pred, gt)
            else:
                wcorrect += self.weighted_ACC_direct(pred, gt)

        wacc = wcorrect / len(answers) * 100
        return wacc
    
class LAMM_Facial_Smile_Classification(Base_Metric):

    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)

    def metric_func(self, answers):
        score = 0.0
        for item in tqdm(answers, desc="Running Metric"):
            gt_label = item['gt']
            pred_text = item['answer']
            text = pred_text.lower()
            words = word_tokenize(text)
            words = [word for word in words if word not in string.punctuation]
            if 'no' in words or 'not' in words:
                pred_label = '-1'
            else:
                pred_label = '1'
            if pred_label == gt_label:
                score += 1.0
            
        return dict(
            ACC = score/len(answers),
        )
        
class LAMM_Facial_Hair_Classification(Base_Metric):

    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)

    def metric_func(self, answers):
        def parse_entity(text):
            text = text.lower()
            words = word_tokenize(text)
            words = [word for word in words if word not in string.punctuation]
            words = [word for word in words if word not in stops]
            words = [wordnet.morphy(word) for word in words if word not in stops]
            return words

        def convert_word(input_word):
            conversion_dict = voc_syndict
            lemmatizer = WordNetLemmatizer()
            input_word = lemmatizer.lemmatize(input_word.lower())
            if input_word in conversion_dict:
                return conversion_dict[input_word]
            else:
                return input_word

        def classification_acc_lamm(gt_text, pred_text):
            convert = False
            if convert:
                pred_text = convert_word(pred_text)
            words = parse_entity(pred_text)
            syn_set = wn.synsets(gt_text)
            try:
                syn_list = syn_set[0].lemma_names() + [gt_text]
            except:
                syn_list = [gt_text]
            if pred_text in syn_list:
                return True
            for syn in syn_list:
                if syn in words:
                    return True
            return False
        score = 0.0
        for item in tqdm(answers, desc="Running Metric"):
            gt_label = item['gt']
            pred_text = item['answer']
            if classification_acc_lamm(gt_label, pred_text):
                score += 1.0
        return dict(
            ACC = score/len(answers)
        )

class LAMM_3D_Classification(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
    
    def metric_func(self, answers):
        score = 0.0
        for item in tqdm(answers, desc="Running Metric"):
            gt_label = item['object_name']
            pred_text = item['answer']
            text = pred_text.lower()
            if gt_label in text:
                score += 1.0
        return dict(
            ACC = score/len(answers)
        )

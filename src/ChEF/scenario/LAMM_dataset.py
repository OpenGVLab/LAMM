import os
import json
from torch.utils.data import Dataset
import random
from .lamm_sysmsg import common_task2sysmsg, locating_task2sysmsg
class FlickrLAMMDataset(Dataset):
    task_name = 'caption_lamm'
    dataset_name = 'Flickr30k'
    
    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Caption_flickr30k.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Caption']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt_answers' : item['sentences'],
        }
        return data_dict
    
class CIFAR10LAMMDataset(Dataset):
    task_name = 'classification_lamm'
    dataset_name = 'CIFAR10'
    CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    def __init__(self, base_data_path, **kwargs):
        super().__init__()
        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Classification_CIFAR10.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Classification']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        random_labels = random.sample(self.CIFAR10_LABELS, len(self.CIFAR10_LABELS))
        shuffled_labels_str = ", ".join(random_labels)

        additional_sentence = "Please choose a label from the following shuffled categories: "

        data_dict = {
            'id' : data_id,
            'label' : item['label'],
            'question' : f"{item['query']} {additional_sentence}{shuffled_labels_str}.",
            'image_path' : os.path.join(self.base_data_path, item['image']),
        }
        return data_dict



class CelebAHairDataset(Dataset):
    task_name = 'Facial_cls_lamm'
    dataset_name = 'CelebA(Hair)'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Facial_Classification_CelebA(Hair).json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Facial_Classification']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)
 
        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt' : item['attr'],
        }
        return data_dict


class CelebASmileDataset(Dataset):
    task_name = 'Facial_cls_lamm'
    dataset_name = 'CelebA(Smile)'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Facial_Classification_CelebA(Smile).json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Facial_Classification']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)
 
        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt' : item['attr'],
        }
        return data_dict
    

class FSC147LAMMDataset(Dataset):
    task_name = 'counting'
    dataset_name = 'FSC147'
    def __init__(self, base_data_path, **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,'meta_file', 'Counting_FSC147.json')
        self.data = json.load(open(json_path,'rb'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['id']) if 'id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,self.data[index]['image']),
            'gt_answers': self.data[index]['num'],
            'question': self.data[index]['query']
        }
        return res_dict
    
class VOC2012LAMMDataset(Dataset):
    task_name = 'detection_lamm'
    dataset_name = 'VOC2012'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Detection_VOC2012.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Detection']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        data_id = str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id': data_id,
            'image_path': os.path.join(self.base_data_path, item['image']),
            'gt_answers': item['object'],
            'question': item['query'],
        }
        return data_dict
    

class SVTDataset(Dataset):
    task_name = 'OCR'
    dataset_name = 'SVT'
    def __init__(self, 
                 base_data_path,
                 **kwargs):
        self.base_data_path = base_data_path
        json_path = os.path.join(base_data_path, 'meta_file', 'OCR_SVT.json')
        self.data = json.load(open(json_path,'rb'))
        self.system_msg = common_task2sysmsg['OCR']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        data_id = str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id': data_id,
            'image_path': os.path.join(self.base_data_path, item['image']),
            'gt_answers': item['word_list'],
            'question': item['query']
        }
        return data_dict

class UCMercedDataset(Dataset):
    task_name = 'lamm_classification'
    dataset_name = 'UCMerced'
    def __init__(self, 
                 base_data_path,
                 **kwargs):
        self.base_data_path = base_data_path
        json_path = os.path.join(base_data_path, 'meta_file', 'Fine-grained_Classification_UCMerced.json')
        self.data = json.load(open(json_path,'rb'))
        self.system_msg = common_task2sysmsg['Fine-grained_Classification']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        data_id = str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id': data_id,
            'image_path': os.path.join(self.base_data_path, item['image']),
            'gt_answers': item['label'],
            'question': item['query']
        }
        return data_dict

class AI2DDataset(Dataset):
    task_name = 'VQA_lamm'
    dataset_name = 'AI2D'
    def __init__(self, 
                 base_data_path,
                 **kwargs):
        self.base_data_path = base_data_path
        json_path = os.path.join(base_data_path, 'meta_file', 'VQA_AI2D.json')
        self.data = json.load(open(json_path,'rb'))
        self.system_msg = common_task2sysmsg['VQA']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt_choice' : item['gt_choice'],
            'gt_choices' : item['gt_choices'],
        }
        return data_dict

class ScienceQALAMMDataset(Dataset):
    task_name = 'VQA_lamm'
    dataset_name = 'ScienceQA'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'VQA_ScienceQA.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['VQA']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt_choice' : item['gt_choice'],
            'gt_choices' : item['gt_choices'],
        }
        return data_dict

class LocatingVOC2012Dataset(Dataset):
    task_name = 'locating'
    dataset_name = 'Locating_VOC2012'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Locating_VOC2012.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = locating_task2sysmsg['VOC2012']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt_answers': item['object']
        }
        return data_dict
    

class LocatingLSPDataset(Dataset):
    task_name = 'locating'
    dataset_name = 'Locating_LSP'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Locating_LSP.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = locating_task2sysmsg['LSP']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id' : data_id,
            'image_path' : os.path.join(self.base_data_path, item['image']),
            'question' : item['query'],
            'gt_answers': item['gt_joints']
        }
        return data_dict

class ScanQALAMMDataset(Dataset):
    task_name = 'VQA_lamm_3D'
    dataset_name = 'ScanQA_LAMM'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'VQA_ScanQA_multiplechoice.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['VQA3D']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id' : data_id,
            'pcl_paths' : os.path.join(self.base_data_path, item['pcl'][2:]),
            'question' : item['query'],
            'gt_choice' : item['gt_choice'],
            'gt_choices' : item['gt_choices'],
        }
        return data_dict
    
class ScanNetLAMMDataset(Dataset):
    task_name = 'Detection_3D'
    dataset_name = 'ScanNet_LAMM'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'Detection_ScanNet.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['Detection3D']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id' : data_id,
            'pcl_paths' : os.path.join(self.base_data_path, item['pcl'][2:]),
            'question' : item['query'],
            'gt_answers': item['object']
        }
        return data_dict
    
class ScanReferLAMMDataset(Dataset):
    task_name = 'VG3D'
    dataset_name = 'ScanRefer_LAMM'

    def __init__(self, base_data_path, **kwargs):
        super().__init__()

        self.base_data_path = base_data_path
        json_path = os.path.join(self.base_data_path, 'meta_file', 'VG_ScanRefer.json')
        self.data = json.load(open(json_path, 'rb'))

        self.system_msg = common_task2sysmsg['VG3D']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id =  str(item['id']) if 'id' in item else str(index)

        data_dict = {
            'id' : data_id,
            'pcl_paths' : os.path.join(self.base_data_path, item['pcl'][2:]),
            'question' : item['query'],
            'gt_answers': item['object']
        }
        return data_dict

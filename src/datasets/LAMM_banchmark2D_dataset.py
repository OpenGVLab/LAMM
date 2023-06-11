import json
import os
from torch.utils.data import Dataset
from PIL import Image
from .system_msg import common_task2sysmsg, locating_task2sysmsg


common_dataset2task = {
    'VOC2012': 'Detection',
    'CIFAR10': 'Classification',
    'SQAimage': 'VQA',
    'SVT':'OCR',
    'flickr30k': 'Caption',
    'FSC147': 'Counting',
    'UCMerced':'Fine-grained_Classification',
    'CelebA(Smile)': 'Facial_Classification',
    'CelebA(Hair)': 'Facial_Classification',
    'AI2D': 'VQA',
    'LSP':'Keypoints_Detection',
}

locating_dataset2task = {
    'VOC2012': 'Locating',
    'LSP':'Locating',
    'FSC147': 'Locating',
}

class LAMM_EVAL_2D(Dataset):
    def __init__(self, 
                 base_data_path,
                 dataset_name,
                 mode = 'common',
                 load_img = True):
        assert mode in ['common','locating']
        self.base_data_path = base_data_path
        self.dataset_name = dataset_name
        if mode == 'common':
            self.task_name = common_dataset2task[self.dataset_name]
            self.system_msg = common_task2sysmsg[self.task_name]
        elif mode == 'locating':
            self.task_name = locating_dataset2task[self.dataset_name]
            self.system_msg = locating_task2sysmsg[self.dataset_name]
        json_path = os.path.join(base_data_path, 'meta_file', self.task_name + '_' + self.dataset_name + '.json')
        self.data = json.load(open(json_path,'rb'))
        self.load_img = load_img
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        if 'image' in data_item:
            img = Image.open(os.path.join(self.base_data_path, data_item['image'])).convert('RGB')
            data_item['image'] = img
        return data_item

    def __repr__(self) -> str:
        repr_str = '{}_{}\n\nSYSTEM_MSG:{}'
        return repr_str.format(self.task_name, self.dataset_name, self.system_msg)


if __name__ == "__main__":
    dataset = LAMM_EVAL_2D(
        base_data_path= 'dataset/LAMM-Dataset/2D_Benchmark',
        dataset_name='SVT',
        mode='common',
        load_img=True
    )
    print(dataset)
    data = dataset[0]
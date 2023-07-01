import json
import os
from torch.utils.data import Dataset
import numpy as np
from .system_msg import common_task2sysmsg, locating_task2sysmsg


common_dataset2task = {
    'ScanNet': 'Detection',
    'ScanRefer': 'VG',
    'ScanQA': 'VQA',
    'ScanQA_multiplechoice': 'VQA',
}


class LAMM_EVAL_3D(Dataset):
    def __init__(self, 
                 base_data_path,
                 dataset_name,
                 mode = 'common',
                 load_data = False):
        assert mode in ['common', 'locating']
        self.base_data_path = base_data_path
        self.dataset_name = dataset_name
        if mode == 'common':
            self.task_name = common_dataset2task[self.dataset_name]
            self.system_msg = common_task2sysmsg[self.task_name + '3D']
        json_path = os.path.join(base_data_path, 'meta_file', self.task_name + '_' + self.dataset_name + '.json')
        self.data = json.load(open(json_path, 'rb'))
        self.load_data = load_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        if 'pcl' in data_item:
            data_item['pcl'] = os.path.join(self.base_data_path, data_item['pcl'])
        return data_item

    def __repr__(self) -> str:
        repr_str = '{}_{}\n\nSYSTEM_MSG:{}'
        return repr_str.format(self.task_name, self.dataset_name, self.system_msg)


if __name__ == "__main__":
    dataset = LAMM_EVAL_3D(
        base_data_path= 'dataset/LAMM-Dataset/3D_Benchmark',
        dataset_name='ScanNet',
        mode='common',
        load_data=False
    )
    print(dataset)
    data = dataset[0]
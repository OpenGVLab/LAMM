import os
import json
from torch.utils.data import Dataset

class HHHDataset(Dataset):
    task_name = 'HHH'
    dataset_name = 'HHH'
    def __init__(self, base_data_path, dimension, **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        meta_base_dir = os.path.join(self.base_data_path, 'meta_file')
        annot_list = os.listdir(meta_base_dir)
        for annot_file in annot_list:
            if annot_file.startswith(dimension):
                self.data = json.load(open(os.path.join(meta_base_dir, annot_file)))
                return
        print(f"Dimension: {dimension} not found.")
        exit()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['id']) if 'id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': [os.path.join(self.base_data_path,img_path) for img_path in self.data[index]['image']],
            'gt_answers': self.data[index]['gt'],
            'question': self.data[index]['query'],
            'source': self.data[index]['source']
        }
        return res_dict
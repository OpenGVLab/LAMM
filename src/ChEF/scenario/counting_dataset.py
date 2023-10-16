import os
import json
from torch.utils.data import Dataset
class FSC147Dataset(Dataset):
    task_name = 'counting'
    dataset_name = 'FSC147'
    def __init__(self, base_data_path = 'data/LAMM/2D_Benchmark/', ppl_cfg = None, **kwargs):
        self.base_data_path = base_data_path
        super().__init__()
        json_path = os.path.join(base_data_path,'meta_file', 'Counting_FSC147.json')
        self.data = json.load(open(json_path,'rb'))
        self.ppl_cfg = ppl_cfg
        if self.ppl_cfg:
            self.heatmap_width = self.ppl_cfg.get('heatmap_width', 2)
            self.load_ppl_options()
    
    def load_ppl_options(self):
        print('----generate ppl negative options----')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        id = str(item['id']) if 'id' in item else str(index)
        res_dict = {
            'id': id,
            'image_path': os.path.join(self.base_data_path,self.data[index]['image']),
            'gt_answers': self.data[index]['num'],
            'question': self.data[index]['query'] + ' Please answer an arabic number directly.'
        }
        if self.ppl_cfg:
            gt_num = res_dict['gt_answers']
            options = []
            object_name = res_dict['question'].replace('How many ', '').replace(' are there in this image?','')
            option_nums = []
            for i in range(max(0, gt_num-self.heatmap_width), gt_num+self.heatmap_width+1):
                option_nums.append(i)
            options = []
            right_pad, left_pad = 1, 1
            for num in option_nums:
                if num // 10 == (gt_num // 10):
                    options.append(str(num))
                elif num // 10 < (gt_num // 10):
                    options.append(str(option_nums[-1] + right_pad))
                    right_pad += 1
                else:
                    options.append(str(option_nums[0] - left_pad))
                    left_pad += 1
            res_dict['options'] = options
        return res_dict
    
if __name__ == '__main__':
    dataset = FSC147Dataset(base_data_path='/mnt/petrelfs/shizhelun/shizhelun/data/datasets/LAMM/2D_Benchmark',
                            ppl_cfg=dict(heatmap_width = 3))
    data = dataset[0]
    import ipdb;ipdb.set_trace()

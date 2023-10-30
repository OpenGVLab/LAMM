from PIL import Image

def copy_batch_dict(batch, idx):
    answer_dict = {}
    for key in batch.keys():
        if not isinstance(batch[key], list):
            answer_dict[key] = batch[key]
        else:
            answer_dict[key] = batch[key][idx]
    if 'image_path' in answer_dict and isinstance(answer_dict['image_path'], Image.Image): # image_path in MMBench is Image.Image
        del answer_dict['image_path']
    return answer_dict
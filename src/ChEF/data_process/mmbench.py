import os
import io
import base64
import pandas as pd
from PIL import Image
def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def main(split='dev'):
    base_path = '../../../data/MMBench'
    save_image_dir = os.path.join(base_path, 'images')
    os.makedirs(save_image_dir,exist_ok=True)
    df = pd.read_csv(os.path.join(base_path, f'mmbench_{split}_20230712.tsv'), sep='\t')
    for i in range(len(df)):
        image = df.iloc[i]['image']
        index = df.iloc[i]['index']
        image = decode_base64_to_image(image)
        image_name = f'mmbench_image_{index}.png'
        image.save(os.path.join(save_image_dir, image_name))


if __name__ == '__main__':
    main()
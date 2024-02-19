import json
import os

from glob import glob
from PIL import Image
import numpy as np

from tqdm.contrib import tzip
from transformers import Pix2StructProcessor

save_root = " " # Save path
os.makedirs(save_root, exist_ok=True)

# You should download the pix2struct-base ckpt as folloing link:
# https://huggingface.co/google/pix2struct-base
input_tokenizer = Pix2StructProcessor.from_pretrained("pix2struct-base")
label_tokenizer = Pix2StructProcessor.from_pretrained("pix2struct-base")
label_tokenizer.image_processor.is_vqa = False
input_tokenizer.image_processor.is_vqa = False

result={}
imgnames = []
imgs = []
texts = []
lens = []
with open('./chart_train.json') as json_file:
    data = json.load(json_file)

for item in data:
    chart_type = item['chart_type']
    imgname = item["imgname"]
    img = item["img"]
    image = Image.open(img)
    topic = item["topic"]
    title = item["title"]
    csv = item["csv"]
    code = item["code"]

    #text = csv + " <title> " + title + " <type> " + chart_type
    text = " <title> " + title + " <type> " + chart_type
    if len(text)<1000:
        imgnames.append(imgname)
        imgs.append(image)
        texts.append(text)
        lens.append(len(text))
print(texts[0])
print(max(lens))

for idx, (name, img, la) in enumerate(tzip(imgnames, imgs, texts)):
    inputs = input_tokenizer(
            images=img,
            #text=random.choice(prompts),
            return_tensors="pt",
            padding="max_length",
            # truncation=True,
            max_patches=input_tokenizer.image_processor.max_patches,
            max_length=1280,
            )

    labels = label_tokenizer(
            text=la, 
            return_tensors="pt", 
            padding="max_length",
            # truncation=True,
            add_special_tokens=True, 
            max_length=1280,
            ).input_ids
               
    np.save(f"{save_root}/{name.split('.')[0]}_{idx}_simv2_input_flattened_patches.npy", inputs.data['flattened_patches'].numpy())
    np.save(f"{save_root}/{name.split('.')[0]}_{idx}_simv2_input_attention_mask.npy", inputs.data['attention_mask'].numpy())
    np.save(f"{save_root}/{name.split('.')[0]}_{idx}_simv2_label.npy", labels.numpy())
        
            
            
        

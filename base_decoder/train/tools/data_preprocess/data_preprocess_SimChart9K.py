import os
import csv
from glob import glob
from PIL import Image
import numpy as np
from random import sample

from tqdm.contrib import tzip
from transformers import Pix2StructProcessor


root = " " # this path needs to cover the root of your dataset
save_root = " " # Save path
os.makedirs(save_root, exist_ok=True)

# You should download the pix2struct-base ckpt as folloing link:
# https://huggingface.co/google/pix2struct-base
input_tokenizer = Pix2StructProcessor.from_pretrained("pix2struct-base") 
label_tokenizer = Pix2StructProcessor.from_pretrained("pix2struct-base")
label_tokenizer.image_processor.is_vqa = False
input_tokenizer.image_processor.is_vqa = False


img_folder_path = os.path.join(root, "png")
tables_folder_path = os.path.join(root, "table")

all_image_name = glob(os.path.join(img_folder_path, "*.png"))
sel_image_name = sample(all_image_name, round(len(all_image_name)*1)) #select part of data
imgnames = []
imgs = []
texts = []
for item in sel_image_name:
    imgname = os.path.split(item)[-1]
    
    image = Image.open(os.path.join(img_folder_path, imgname))
    table_path = os.path.join(tables_folder_path, imgname.replace(".png", ".csv"))
    text = ""
    with open(table_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:  # Iterate through the loop to read line by line
            text = text + " \\t ".join(line) + " \\n "
            
    imgnames.append(imgname)
    imgs.append(image)
    texts.append(text)

for idx, (name, img, la) in enumerate(tzip(imgnames, imgs, texts)):
    inputs = input_tokenizer(
            images=img,
            return_tensors="pt",
            padding="max_length",
            max_patches=input_tokenizer.image_processor.max_patches,
            max_length=1280,
            )

    labels = label_tokenizer(
            text=la, 
            return_tensors="pt", 
            padding="max_length",
            add_special_tokens=True, 
            max_length=1280,
        ).input_ids
            
    np.save(f"{save_root}/{name.split('.')[0]}_{idx}_simchart9k_input_flattened_patches.npy", inputs.data['flattened_patches'].numpy())
    np.save(f"{save_root}/{name.split('.')[0]}_{idx}_simchart9k_input_attention_mask.npy", inputs.data['attention_mask'].numpy())
    np.save(f"{save_root}/{name.split('.')[0]}_{idx}_simchart9k_label.npy", labels.numpy())

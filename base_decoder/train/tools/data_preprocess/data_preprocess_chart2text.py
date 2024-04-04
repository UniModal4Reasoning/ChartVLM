import os
import csv
from glob import glob
from PIL import Image
import numpy as np
import torch
from tqdm.contrib import tzip
from transformers import Pix2StructProcessor
import random
from random import sample
import json
from tqdm import tqdm

splits = ["images", "multiColumn"]
subsets = ["matplot", "statista"]  # "human", 
root = " " # this path needs to cover the root of your dataset
save_root = " " # Save path
os.makedirs(save_root, exist_ok=True)


input_tokenizer = Pix2StructProcessor.from_pretrained("pix2struct-base") 
label_tokenizer = Pix2StructProcessor.from_pretrained("pix2struct-base")
label_tokenizer.image_processor.is_vqa = False
input_tokenizer.image_processor.is_vqa = False

for split in tqdm(splits):
    for subset in subsets:

        img_folder_path = os.path.join(root, split, subset)
        tables_folder_path = os.path.join(root, split, "table")
        
        all_image_name = glob(os.path.join(img_folder_path, "*.png"))
        sel_image_name = sample(all_image_name, round(len(all_image_name)*1))
        imgnames = []
        imgs = []
        texts = []
        for item in tqdm(sel_image_name):
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
                
            np.save(f"{save_root}/{name.split('.')[0]}_{idx}_chart2text_input_flattened_patches.npy", inputs.data['flattened_patches'].numpy())
            np.save(f"{save_root}/{name.split('.')[0]}_{idx}_chart2text_input_attention_mask.npy", inputs.data['attention_mask'].numpy())
            np.save(f"{save_root}/{name.split('.')[0]}_{idx}_chart2text_label.npy", labels.numpy())
            
                
                
            
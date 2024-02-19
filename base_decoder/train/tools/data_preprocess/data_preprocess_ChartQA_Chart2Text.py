import os
import csv
from glob import glob
from PIL import Image
import numpy as np
from random import sample

from tqdm.contrib import tzip
from transformers import Pix2StructProcessor

splits = ["train", "val"]
subset = ["augmented", "human"]  # "human", 
root = " " # this path needs to cover the root of your dataset
save_root = " " # Save path
os.makedirs(save_root, exist_ok=True)

# You should download the pix2struct-base ckpt as folloing link:
# https://huggingface.co/google/pix2struct-base
input_tokenizer = Pix2StructProcessor.from_pretrained("pix2struct-base") 
label_tokenizer = Pix2StructProcessor.from_pretrained("pix2struct-base")
label_tokenizer.image_processor.is_vqa = False
input_tokenizer.image_processor.is_vqa = False
"""
prompts = ['Can you create a table based on the data in the chart below?',
        'I need you to generate a table using the underlying data in the following figure.',
        'Please produce a table using the data presented in the chart below.',
        'From the figure presented below, can you generate a table with the underlying data?',
        'I would like you to create a table using the data that underlies the chart below.',
        'Please help me generate a table based on the data underlying the figure presented below.',
        'Can you prepare a table utilizing the data from the chart below?',
        'Using the data presented in the chart below, can you create a table?',
        'Kindly generate a table with the underlying data from the figure presented below.',
        'The table that shows the data underlying the following chart needs to be generated, can you do it?']
"""

for split in splits:
    
    sub_split_save_root = os.path.join(save_root, split)
    os.makedirs(sub_split_save_root, exist_ok=True)
    
    img_folder_path = os.path.join(root, split, "png")
    tables_folder_path = os.path.join(root, split, "table")
    
    all_image_name = glob(os.path.join(img_folder_path, "*.png"))
    sel_image_name = sample(all_image_name, round(len(all_image_name)*1))
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
    print(len(texts))
    print(len(imgs))
    print(texts[10])
    length = []
    for idx, (name, img, la) in enumerate(tzip(imgnames, imgs, texts)):
        # length.append(len(la))
    # print(max(length))  # 1814  1426  1143
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
               
        np.save(f"{sub_split_save_root}/{name.split('.')[0]}_{idx}_sim_input_flattened_patches.npy", inputs.data['flattened_patches'].numpy())
        np.save(f"{sub_split_save_root}/{name.split('.')[0]}_{idx}_sim_input_attention_mask.npy", inputs.data['attention_mask'].numpy())
        np.save(f"{sub_split_save_root}/{name.split('.')[0]}_{idx}_sim_label.npy", labels.numpy())
        
            
            
        
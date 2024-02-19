import os
import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

def infer_base_decoder(image, model_path, max_token=1280, title_type=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if title_type == False:
        base_decoder = Pix2StructForConditionalGeneration.from_pretrained(os.path.join(model_path,'base_decoder'))
    if title_type == True:
        base_decoder = Pix2StructForConditionalGeneration.from_pretrained(os.path.join(model_path,'base_decoder','title_type'))
    processor_base_decoder = Pix2StructProcessor.from_pretrained(os.path.join(model_path,'base_decoder'))
    processor_base_decoder.image_processor.is_vqa = False
    base_decoder.to(device)

    inputs_base_decoder = processor_base_decoder(images=image, return_tensors="pt")
    inputs_base_decoder = inputs_base_decoder.to(device)

    predictions_base_decoder = base_decoder.generate(**inputs_base_decoder, max_new_tokens = max_token)

    output_base_decoder = processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)

    return output_base_decoder
import os
import argparse
import torch
import json
import time 
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, '../')
from tools.ChartVLM import infer_ChartVLM

if __name__ == "__main__":
    os.makedirs(os.path.join("infer_result"), exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the training data file")
    parser.add_argument("--benchmark_json_path", required=True, help="Path to save the trained model")
    args = parser.parse_args()
    model = args.model_path
    benchmark_json = args.benchmark_json_path

    results = []
    with open(benchmark_json) as json_file:
        data = json.load(json_file)

    for item in tqdm(data):
        chart_type = item['chart_type']
        imgname = item["imgname"]
        topic = item["topic"]
        title = item["title"]
        csv = item["csv"]
        img = item["img"].replace('./','ChartX/')

        question_qa = item['QA']['input'] 

        output_title = infer_ChartVLM(img, 'What is the title of the chart image?', model)
        output_type = infer_ChartVLM(img, 'What is the chart type of the image?', model)
        output_csv = infer_ChartVLM(img, 'Convert the chart image to CSV.', model)
        output_des = infer_ChartVLM(img, 'Generate a descriptive text according to the chart image.', model)
        output_sum = infer_ChartVLM(img, 'Create a brief summarization or extract key insights based on the chart image.', model)
        output_redraw = infer_ChartVLM(img, 'Redraw the chart image using Python code.', model)
        output_qa = infer_ChartVLM(img, question_qa, model)
        
        result = {"imgname":imgname,"title_gt":title, "title_pred":output_title, "type_gt": chart_type, "type_pred":output_type, "csv_gt":csv, "csv_pred":output_csv, "description_pred":output_des, "summarization_pred":output_sum, "redrawing_code_pred":output_redraw, "QA":{"question":question_qa, "answer_gt":item['QA']['output'], 'answer_pred':output_qa}}            
        results.append(result)
    
    with open(os.path.join("infer_result", "infer_result_on_ChartX.json"), "w", encoding='utf-8') as f:  
        json.dump(results, f, indent=2, ensure_ascii=False)
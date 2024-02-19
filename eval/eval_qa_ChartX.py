# Evaluation script for question-answering task in ChartX
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Hancheng Ye, Renqiu Xia 
# All Rights Reserved 2024-2025.

import os
import json
import time
import logging
import argparse
import datetime
from tqdm import tqdm

from metric.gpt_acc import eval_gpt_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_result_dir", required=True, help="Path to the inference result data")
    parser.add_argument("--your_openai_key", required=True, help="Path to the inference result data")
    args = parser.parse_args()
    infer_result = args.infer_result_dir
    openai_key = args.your_openai_key
   

    len_sum = 0

    easy_types = ['bar_chart', 'line_chart', 'pie_chart', 'bar_chart_num', 'line_chart_num', 'rings',  'heatmap',  'box', 'candlestick', 'funnel', 'histogram', 'treemap']
    diff_types = ['rose', 'area_chart','3D-Bar','bubble','multi-axes', 'radar']
    single_class_chart = ['histogram', 'rose', 'rings', 'funnel', 'pie', 'treemap']
    chart_types = ['bar_chart', 'line_chart', 'pie_chart', 'bar_chart_num', 'line_chart_num', 'rings', 'rose', 'area_chart', 'heatmap', '3D-Bar', 'box', 'bubble', 'candlestick', 'funnel', 'histogram', 'multi-axes', 'radar', 'treemap']

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"eval_result_qa_on_ChartX_{current_time}.log"
    os.makedirs(os.path.join("eval_result",'qa'), exist_ok=True)
    logging.basicConfig(filename=os.path.join("eval_result",'qa', log_file), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('infer_result:'+ infer_result)

    qa_score_set_total = []
    for c in tqdm(chart_types):
        with open(infer_result) as json_file:  
            data = json.load(json_file)

        qa_score_set = []

        for item in data:
            chart_type = item['type_gt']
            title = item["title_gt"]
            #imgname = item["imgname"]
            csv_gt = item["csv_gt"]
            question = item["QA"]['question']
            answer_gt = item["QA"]['answer_gt']
            answer_pred = item["QA"]['answer_pred']

        
            if chart_type == c:                
                qa_score = eval_gpt_acc(question, answer_gt, answer_pred, openai_key)
                qa_score_set.append(qa_score)
                qa_score_set_total.append(qa_score)      
        qa_score_type = sum(qa_score_set)/len(qa_score_set)
        logging.info('*************** Performance *****************')
        logging.info(c)
        logging.info('%.4f' % qa_score_type)


    qa_score_total = sum(qa_score_set_total)/len(qa_score_set_total)
    logging.info('*************** Performance *****************')
    logging.info('average')
    logging.info('%.4f' % qa_score_total)
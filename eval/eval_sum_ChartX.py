# Evaluation script for summarization task in ChartX
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Hancheng Ye, Renqiu Xia 
# All Rights Reserved 2024-2025.

import os
import argparse
import json
import time
import logging
import datetime
from tqdm import tqdm

from metric.gpt_score import eval_gpt_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_result_dir", required=True, help="Path to the inference result data")
    parser.add_argument("--your_openai_key", required=True, help="Path to the inference result data")
    args = parser.parse_args()
    infer_result = args.infer_result_dir
    openai_key = args.your_openai_key
   

    results = []
    len_sum = 0

    easy_types = ['bar_chart', 'line_chart', 'pie_chart', 'bar_chart_num', 'line_chart_num', 'rings',  'heatmap',  'box', 'candlestick', 'funnel', 'histogram', 'treemap']
    diff_types = ['rose', 'area_chart','3D-Bar','bubble','multi-axes', 'radar']
    single_class_chart = ['histogram', 'rose', 'rings', 'funnel', 'pie', 'treemap']
    chart_types = ['bar_chart', 'line_chart', 'pie_chart', 'bar_chart_num', 'line_chart_num', 'rings', 'rose', 'area_chart', 'heatmap', '3D-Bar', 'box', 'bubble', 'candlestick', 'funnel', 'histogram', 'multi-axes', 'radar', 'treemap']

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"eval_result_sum_on_ChartX_{current_time}.log"
    os.makedirs(os.path.join("eval_result",'sum'), exist_ok=True)
    logging.basicConfig(filename=os.path.join("eval_result",'sum', log_file), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('infer_result:'+ infer_result)


    criterion = f"""
        You're an expert evaluating a model's summarization of a chart, based on its alignment with the ground truth and raw data. Score the model from 0 to 5 based on these criteria:
        0 points: The summary is irrelevant or shows no understanding of the original text, failing to address the core content or theme.
        1 point: While referencing the original text, the summary contains predominantly incorrect details or interpretations, showing minimal understanding and significant inaccuracies.
        2 points: The summary captures some correct details, indicating a basic understanding. However, it misses key elements or includes major inaccuracies, leading to a flawed interpretation of the text.
        3 points: Most details in the summary are accurate, reflecting a good understanding of the original text. Minor errors or omissions are present but don't significantly impact the overall accuracy or comprehension.
        4 points: The summary accurately represents all main ideas and important details of the original text. It shows a very good understanding, with minor room for improvement in clarity, conciseness, or structure.
        5 points: This represents a comprehensive and accurate summary, perfectly encapsulating all essential aspects of the original text. It demonstrates excellent understanding, is error-free, clear, concise, well-structured, and serves as an excellent standalone representation of the original content.
        Score the model's summarization on this scale, providing a single value without providing any reasons.
        """

    summ_score_set_total = []
    for c in tqdm(chart_types):
        with open(infer_result) as json_file:  
            data = json.load(json_file)

        summ_score_set = []

        for item in data:
            chart_type = item['type_gt']
            title = item["title_gt"]
            #imgname = item["imgname"]
            csv_gt = item["csv_gt"]
            summ = item["summarization_pred"]


            content = f'''
            data: {csv_gt} <title> {title} <type> {chart_type}\n
            summarization: {summ}
            '''
        
            if chart_type == c:                
                summ_score = eval_gpt_score(content, criterion, openai_key)
                summ_score_set.append(summ_score)
                summ_score_set_total.append(summ_score)       
        summ_score_type = sum(summ_score_set)/len(summ_score_set)
        logging.info('*************** Performance *****************')
        logging.info(c)
        logging.info('%.4f' % summ_score_type)


    summ_score_total = sum(summ_score_set_total)/len(summ_score_set_total)
    logging.info('*************** Performance *****************')
    logging.info('average')
    logging.info('%.4f' % summ_score_total)
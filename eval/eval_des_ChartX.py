# Evaluation script for description task in ChartX
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
    log_file = f"eval_result_des_on_ChartX_{current_time}.log"
    os.makedirs(os.path.join("eval_result",'des'), exist_ok=True)
    logging.basicConfig(filename=os.path.join("eval_result",'des', log_file), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('infer_result:'+ infer_result)


    criterion = f"""
        You're an expert evaluating a model's description of a chart, based on its alignment with the ground truth and raw data. Score the model from 0 to 5 based on these criteria:
        0 points: Description irrelevant or shows no understanding of the chart or data.
        1 point: Refers to the chart but with largely incorrect details; minimal understanding.
        2 points: Some correct details, but key elements are wrong or missing; basic understanding with significant errors.
        3 points: Most details are correct; good understanding but with minor errors/omissions.
        4 points: All details are correct; very good understanding, minor improvements possible.
        5 points: Comprehensive, accurate description; excellent understanding with no errors; clear and detailed, perfect as a standalone explanation.
        Score the model's description on this scale, providing a single value without providing any reasons.
        """

    des_score_set_total = []
    for c in tqdm(chart_types):
        with open(infer_result) as json_file:  
            data = json.load(json_file)

        des_score_set = []

        for item in data:
            chart_type = item['type_gt']
            title = item["title_gt"]
            #imgname = item["imgname"]
            csv_gt = item["csv_gt"]
            des = item["description_pred"]


            content = f'''
            data: {csv_gt} <title> {title} <type> {chart_type}\n
            description: {des}
            '''
        
            if chart_type == c:                
                des_score = eval_gpt_score(content, criterion, openai_key)
                des_score_set.append(des_score)
                des_score_set_total.append(des_score)       
        des_score_type = sum(des_score_set)/len(des_score_set)
        logging.info('*************** Performance *****************')
        logging.info(c)
        logging.info('%.4f' % des_score_type)


    des_score_total = sum(des_score_set_total)/len(des_score_set_total)
    logging.info('*************** Performance *****************')
    logging.info('average')
    logging.info('%.4f' % des_score_total)
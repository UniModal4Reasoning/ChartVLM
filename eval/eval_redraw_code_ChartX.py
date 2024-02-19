# Evaluation script for redrawing task in ChartX
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
    log_file = f"eval_result_redrawing_on_ChartX_{current_time}.log"
    os.makedirs(os.path.join("eval_result",'redrawing'), exist_ok=True)
    logging.basicConfig(filename=os.path.join("eval_result",'redrawing', log_file), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('infer_result:'+ infer_result)


    criterion = f"""
        You're an expert evaluating a redrawing code of a chart, based on its alignment with the ground truth and raw data. Score the code from 0 to 5 based on these criteria:
        0 points: Completely Irrelevant or Non-functional Code. Demonstrates no understanding of the chart structure or data. Code is inexecutable or produces a completely unrelated chart.
        1 point: Attempted Redraw with Major Discrepancies. Partial understanding of basic chart structure or data. Generated chart has very little in common with the original.
        2 points: Partially Correct Code with Key Errors or Omissions. Basic understanding of chart structure or data but with significant errors. Chart somewhat resembles the original but key inaccuracies are evident.
        3 points: Mostly Accurate; Good Understanding with Minor Errors/Omissions. Accurately reflects most of the chart's structure and data. Generated chart is similar to the original but has a few minor errors.
        4 points: Highly Accurate; Very Good Understanding with Minor Room for Improvement. Accurately presents the chart's structure and data in full. Generated chart is very close to the original, with negligible differences.
        5 points: Comprehensive, Accurate Code; Excellent Understanding, No Errors. Perfectly replicates all details and data of the chart. Generated chart is indistinguishable from the original, flawless.
        Score the redrawing code on this scale, providing a single value without providing ant reasons.
        """

    redrawing_score_set_total = []
    for c in tqdm(chart_types):
        with open(infer_result) as json_file:  
            data = json.load(json_file)

        redrawing_score_set = []

        for item in data:
            chart_type = item['type_gt']
            title = item["title_gt"]
            #imgname = item["imgname"]
            csv_gt = item["csv_gt"]
            redrawing = item["redrawing_code_pred"]


            content = f'''
            data: {csv_gt} <title> {title} <type> {chart_type}\n
            redrawing code: {redrawing}
            '''
        
            if chart_type == c:                
                redrawing_score = eval_gpt_score(content, criterion, openai_key)
                redrawing_score_set.append(redrawing_score)
                redrawing_score_set_total.append(redrawing_score)       
        redrawing_score_type = sum(redrawing_score_set)/len(redrawing_score_set)
        logging.info('*************** Performance *****************')
        logging.info(c)
        logging.info('%.4f' % redrawing_score_type)


    redrawing_score_total = sum(redrawing_score_set_total)/len(redrawing_score_set_total)
    logging.info('*************** Performance *****************')
    logging.info('average')
    logging.info('%.4f' % redrawing_score_total)
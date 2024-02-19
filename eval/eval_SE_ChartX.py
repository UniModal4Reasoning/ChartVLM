# Evaluation script for structural extraction task in ChartX
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Renqiu Xia, Hancheng Ye
# All Rights Reserved 2024-2025.

import os
import argparse
import json
import time
import logging
import datetime
from tqdm import tqdm

from metric.SCRM import csv_eval, draw_SCRM_table



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_result_dir", required=True, help="Path to the inference result data")
    args = parser.parse_args()
    infer_result = args.infer_result_dir

    results = {}
    len_sum = 0

    easy_types = ['bar_chart', 'line_chart', 'pie_chart', 'bar_chart_num', 'line_chart_num', 'rings',  'heatmap',  'box', 'candlestick', 'funnel', 'histogram', 'treemap']
    diff_types = ['rose', 'area_chart','3D-Bar','bubble','multi-axes', 'radar']
    single_class_chart = ['histogram', 'rose', 'rings', 'funnel', 'pie', 'treemap']
    chart_types = ['bar_chart', 'line_chart', 'pie_chart', 'bar_chart_num', 'line_chart_num', 'rings', 'rose', 'area_chart', 'heatmap', '3D-Bar', 'box', 'bubble', 'candlestick', 'funnel', 'histogram', 'multi-axes', 'radar', 'treemap']

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"eval_result_SE_on_ChartX_{current_time}.log"
    os.makedirs(os.path.join("eval_result",'SE'), exist_ok=True)
    logging.basicConfig(filename=os.path.join("eval_result",'SE',log_file), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('infer_result:'+ infer_result)

    for c in tqdm(chart_types):
        with open(infer_result) as json_file:  
            data = json.load(json_file)

        csv_gt_set = []
        csv_pred_set = []

        for item in data:
            chart_type = item['type_gt']
            title = item["title_gt"]
            imgname = item["imgname"]
            csv_gt = item["csv_gt"]
            csv_pred = item["csv_pred"]

            if chart_type in single_class_chart:      #entity replace
                csv_gt = csv_gt.split("\\n")[0].split("\\t")[0] + "\\t " + "value" + " \\n" + csv_gt[csv_gt.index("\\n")+2:]
                try:
                    csv_pred = csv_pred.split("\\n")[0].split("\\t")[0] + "\\t " + "value" + " \\n" + csv_pred[csv_pred.index("\\n")+2:]
                except ValueError:
                    csv_pred = csv_pred
        
            if chart_type == c:                
                csv_gt_set.append(csv_gt)
                csv_pred_set.append(csv_pred)

        if c in easy_types:
            easy = 1
        else:
            easy = 0
        
        len_sum = len_sum + len(csv_pred_set)
        em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high = csv_eval(csv_pred_set, csv_gt_set, easy)
        result = {c:{"value": [em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high], "len":len(csv_gt_set)}}
        results.update(result)

        result_table = draw_SCRM_table(em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high)
        logging.info('*************** Performance *****************')
        logging.info(c)
        logging.info('\n'+ result_table)

    #total SCRM    
    em = 0
    map_strict = 0
    map_slight = 0
    map_high = 0
    ap_50_strict = 0
    ap_75_strict = 0
    ap_90_strict = 0
    ap_50_slight = 0
    ap_75_slight = 0
    ap_90_slight = 0
    ap_50_high = 0
    ap_75_high =0
    ap_90_high = 0
    for k,v in results.items():
        em_s = results[k]["value"][0]*results[k]["len"]/len_sum
        em = em + em_s
        map_strict_s = results[k]["value"][1]*results[k]["len"]/len_sum
        map_strict =  map_strict + map_strict_s
        map_slight_s = results[k]["value"][2]*results[k]["len"]/len_sum
        map_slight = map_slight + map_slight_s
        map_high_s = results[k]["value"][3]*results[k]["len"]/len_sum
        map_high = map_high + map_high_s
        ap_50_strict_s = results[k]["value"][4]*results[k]["len"]/len_sum
        ap_50_strict = ap_50_strict + ap_50_strict_s
        ap_75_strict_s = results[k]["value"][5]*results[k]["len"]/len_sum
        ap_75_strict = ap_75_strict + ap_75_strict_s
        ap_90_strict_s = results[k]["value"][6]*results[k]["len"]/len_sum
        ap_90_strict = ap_90_strict + ap_90_strict_s
        ap_50_slight_s = results[k]["value"][7]*results[k]["len"]/len_sum
        ap_50_slight = ap_50_slight + ap_50_slight_s
        ap_75_slight_s = results[k]["value"][8]*results[k]["len"]/len_sum
        ap_75_slight = ap_75_slight + ap_75_slight_s
        ap_90_slight_s = results[k]["value"][9]*results[k]["len"]/len_sum
        ap_90_slight = ap_90_slight + ap_90_slight_s
        ap_50_high_s = results[k]["value"][10]*results[k]["len"]/len_sum
        ap_50_high = ap_50_high + ap_50_high_s
        ap_75_high_s = results[k]["value"][11]*results[k]["len"]/len_sum
        ap_75_high = ap_75_high + ap_75_high_s
        ap_90_high_s = results[k]["value"][12]*results[k]["len"]/len_sum
        ap_90_high = ap_90_high + ap_90_high_s
        
        
    result_table_total = draw_SCRM_table(em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high)
    logging.info('*************** Performance *****************')
    logging.info("average")
    logging.info('\n'+ result_table_total)


# Evaluation process of image-to-text task using Pix2Struct Model
# Reference https://arxiv.org/abs/2210.03347 and https://arxiv.org/abs/2309.11268
# Written by Bo Zhang, Renqiu Xia, Haoyang Peng
# All Rights Reserved 2024-2025.

import time

import numpy as np
import torch
import tqdm

from p2s.utils import common_utils, commu_utils
from p2s.models import load_data_to_gpu
from p2s.utils import common_utils, commu_utils
from transformers import Pix2StructForConditionalGeneration


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


@torch.no_grad()
def eval_one_epoch(model: Pix2StructForConditionalGeneration, tokenizer, val_loader,
                   criterion, rank, dist_test=False, relaxed_accuracy=0., max_num = 200):

    if rank == 0:
        pbar = tqdm.tqdm(total=len(val_loader), leave=True, desc='eval', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    targets = []
    predictions = []
    model.eval()

    end = time.time()
    for i, batch in enumerate(val_loader):
        
        data_timer = time.time()
        cur_data_time = data_timer - end                                                                                                                                   
        
        net_device = next(model.parameters()).device

        labels = batch.pop("labels").to(net_device)
        inputs = {
            "flattened_patches": batch.pop("flattened_patches").to(net_device),
            "attention_mask": batch.pop("attention_mask").to(net_device)
            }

        if dist_test:
            output_ids = model.module.generate(**inputs, max_new_tokens=max_num)       
        else:
            output_ids = model.generate(**inputs, max_new_tokens=max_num)
        
        pred_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
        #print(pred_text)
        #print(labels_text)

        predictions.extend(pred_text)
        targets.extend(labels_text)
        
        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict = {
                'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            }
            pbar.set_postfix(disp_dict)
            pbar.update()

        end = time.time()
        
    if rank == 0:
        pbar.close()

    em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high = criterion(
        predictions=predictions,
        references=targets
        )
    avg_em = commu_utils.average_reduce_value(em)
    avg_map_strict = commu_utils.average_reduce_value(map_strict)
    avg_map_slight = commu_utils.average_reduce_value(map_slight)
    avg_map_high = commu_utils.average_reduce_value(map_high)
    avg_ap_50_strict = commu_utils.average_reduce_value(ap_50_strict)
    avg_ap_75_strict = commu_utils.average_reduce_value(ap_75_strict)
    avg_ap_90_strict = commu_utils.average_reduce_value(ap_90_strict)
    avg_ap_50_slight = commu_utils.average_reduce_value(ap_50_slight)
    avg_ap_75_slight = commu_utils.average_reduce_value(ap_75_slight)
    avg_ap_90_slight = commu_utils.average_reduce_value(ap_90_slight)
    avg_ap_50_high = commu_utils.average_reduce_value(ap_50_high)
    avg_ap_75_high = commu_utils.average_reduce_value(ap_75_high)
    avg_ap_90_high = commu_utils.average_reduce_value(ap_90_high)

    return avg_em, avg_map_strict, avg_map_slight, avg_map_high, avg_ap_50_strict, avg_ap_75_strict, avg_ap_90_strict, avg_ap_50_slight, avg_ap_75_slight, avg_ap_90_slight, avg_ap_50_high, avg_ap_75_high, avg_ap_90_high
    


if __name__ == '__main__':
    pass

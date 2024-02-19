import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from eval_utils import eval_utils

from p2s.datasets import build_dataloader
from p2s.utils import common_utils, metrics_evaluate
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from p2s.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, AutoProcessor, AutoTokenizer



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config', type=str, default="", help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, 
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--num_token', type=int, default=200, help='specify the max number of output tokens for evaluation')
    parser.add_argument('--criterion', type=str, default="exact_match", help='specify the metrics for evaluation')


    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    cfg.TAG = Path(args.config).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.config.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(logger, model, tokenizer, test_loader, criterion, start_epoch,
                     rank, dist_test=False, relaxed_accuracy=0., max_num=200):
    print("******model for testing",model)
    # start evaluation
    em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high = eval_utils.eval_one_epoch(
        model, tokenizer, test_loader, criterion, rank, dist_test=dist_test, max_num=max_num)
    logger.info('*************** Performance of current EPOCH %s *****************' % start_epoch)
    logger.info('\n'+ 
    '-----------------------------------------------------------'+'\n'+
    '|  Metrics   |  Sim_threshold  |  Tolerance  |    Value   |'+'\n'+
    '-----------------------------------------------------------'+'\n'+
    '|            |                 |   strict    |   %.4f     |' % map_strict + '\n'+
    '|            |                 ----------------------------'+'\n'+
    '| mPrecison  |  0.5:0.05:0.95  |   slight    |   %.4f     |' % map_slight + '\n'+
    '|            |                  ---------------------------'+'\n'+
    '|            |                 |    high     |   %.4f     |' % map_high + '\n'+
    '-----------------------------------------------------------'+'\n'+
    '|            |                 |   strict    |   %.4f     |' % ap_50_strict + '\n'+
    '|            |                  ---------------------------'+'\n'+
    '| Precison   |       0.5       |   slight    |   %.4f     |' % ap_50_slight + '\n'+
    '|            |                  ---------------------------'+'\n'+
    '|            |                 |    high     |   %.4f     |' % ap_50_high + '\n'+
    '-----------------------------------------------------------'+'\n'+
    '|            |                 |   strict    |   %.4f     |' % ap_75_strict + '\n'+
    '|            |                  ---------------------------'+'\n'+
    '| Precison   |      0.75       |   slight    |   %.4f     |' % ap_75_slight + '\n'+
    '|            |                  ---------------------------'+'\n'+
    '|            |                 |    high     |   %.4f     |' % ap_75_high + '\n'+
    '-----------------------------------------------------------'+'\n'+
    '|            |                 |   strict    |   %.4f     |' % ap_90_strict + '\n'+
    '|            |                  ---------------------------'+'\n'+
    '| Precison   |       0.9       |   slight    |   %.4f     |' % ap_90_slight + '\n'+
    '|            |                  ---------------------------'+'\n'+
    '|            |                 |    high     |   %.4f     |' % ap_90_high + '\n'+
    '-----------------------------------------------------------'+'\n'+
    '|Precison(EM)|        1        |   strict    |   %.4f     |' % em + '\n'+
    '-----------------------------------------------------------'+'\n'

)

def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continues

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(logger, model, tokenizer, test_loader, criterion, ckpt_dir, eval_output_dir, args, dist_test=False, relaxed_accuracy=0., max_num=200):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.from_pretrained(filename=cur_ckpt)

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            model, tokenizer, test_loader, criterion, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file, max_num=max_num
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    start_epoch = 0
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    # Loading the epoch number from the opt_dict
    if args.ckpt:
        if cfg.MODEL.SAVE_MODE == "huggingface":
            assert os.path.isdir(args.ckpt), f"{args.ckpt} is not a dictionary!"
                       
            state_dict = torch.load(f"{args.ckpt}/state_dict.pth", map_location="cpu")
            start_epoch = state_dict['epoch']
        else:
            #raise NotImplementedError
            start_epoch = 0
    
    if not args.eval_all:
        epoch_id = start_epoch
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / 'val'
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    logger.info('GPU_NAME=%s' % torch.cuda.get_device_name())

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    _, test_loader, test_sampler = build_dataloader(
        mission_name=cfg.MISSION.MISSION_NAME, 
        data_root=cfg.DATA.DATA_ROOT, 
        subset=cfg.DATA.SUB_SET, 
        max_patches=cfg.DATA.MAX_PATCHES, 
        batch_size=args.batch_size, 
        dist=dist_test,
        workers=args.workers, 
        training=False
    )   

    if args.ckpt and cfg.MODEL.SAVE_MODE == "huggingface":
        model = Pix2StructForConditionalGeneration.from_pretrained(args.ckpt)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('NUM_PARAMS=%s' % n_parameters)
    else:
    #    raise NotImplementedError
        model = Pix2StructForConditionalGeneration.from_pretrained(cfg.MODEL.PRETRAIN_MODEL_PATH)
    labels_tokenizer = Pix2StructProcessor.from_pretrained(cfg.PROCESSOR.PROCESSOR_NAME)    
    criterion = metrics_evaluate.Metrics(args.criterion)
    
    model.cuda()
    if dist_test:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(logger, model, labels_tokenizer, test_loader, criterion, ckpt_dir, eval_output_dir, args, dist_test=dist_test, max_num=args.num_token)
        else:
            eval_single_ckpt(logger, model, labels_tokenizer, test_loader, criterion, start_epoch, cfg.LOCAL_RANK, dist_test=dist_test, max_num=args.num_token)


if __name__ == '__main__':
    main()

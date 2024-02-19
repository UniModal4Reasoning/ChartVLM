# print('program started')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # filter the pytorch userwarning
import os

import glob
import _init_path
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from p2s.datasets import build_dataloader, build_dataloader_multi_db
from p2s.utils import common_utils, metrics_evaluate
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from p2s.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config', type=str, default="", help='specify the config for training')
    
    parser.add_argument('--VAL_PER_EPOCH', type=int, default=1, help='perform the evaluation per epoch')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
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


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('********************** Start logging **********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
        
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.config, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create network & dataloader & optimizer---------------------------
    if args.ckpt and cfg.MODEL.SAVE_MODE == "huggingface":
        model = Pix2StructForConditionalGeneration.from_pretrained(args.ckpt)
    else:
        model = Pix2StructForConditionalGeneration.from_pretrained(cfg.MODEL.PRETRAIN_MODEL_PATH)
    labels_tokenizer = Pix2StructProcessor.from_pretrained(cfg.PROCESSOR.PROCESSOR_NAME)    
    
    logger.info(model)
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    
    train_set, _, train_loader, train_sampler = build_dataloader_multi_db(
        mission_name=cfg.MISSION.MISSION_NAME, 
        data_root_1=cfg.DATA.DATA_ROOT_1,
        data_root_2=cfg.DATA.DATA_ROOT_2, 
        subset=cfg.DATA.SUB_SET, 
        max_patches=cfg.DATA.MAX_PATCHES, 
        batch_size=args.batch_size, 
        dist=dist_train,
        workers=args.workers, 
        training=True
    )
    
    _, val_loader, _ = build_dataloader(
        mission_name=cfg.MISSION.MISSION_NAME, 
        data_root=cfg.DATA.DATA_ROOT_1, 
        subset=cfg.DATA.SUB_SET, 
        max_patches=cfg.DATA.MAX_PATCHES, 
        batch_size=args.batch_size, 
        dist=dist_train,
        workers=args.workers, 
        training=False
    )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('NUM_PARAMS=%s' % n_parameters)
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    criterion = metrics_evaluate.Metrics(args.criterion)
    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1

    if args.ckpt:
        if cfg.MODEL.SAVE_MODE == "huggingface":
            assert os.path.isdir(args.ckpt), f"{args.ckpt} is not a dictionary!"
                       
            state_dict = torch.load(f"{args.ckpt}/state_dict.pth", map_location="cpu")
            start_epoch = state_dict['epoch']
            optimizer.load_state_dict(state_dict['optimizer'])
            # best_acc = float(state_dict['best_acc']) # Not save best_acc in state_dict
            it = int(state_dict["it"])
        
        elif cfg.MODEL.SAVE_MODE == "pytorch":
            assert os.path.isfile(args.ckpt), f"{args.ckpt} is not a file!"
            
            checkpoint = torch.load(args.ckpt, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = float(checkpoint['best_acc'])
            it = int(checkpoint["it"])
        else:
            raise NotImplementedError
        
        last_epoch = start_epoch + 1


    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info(f'********************** Start training {cfg.EXP_GROUP_PATH} / {cfg.TAG}({args.extra_tag}) **********************')
    
    train_model(
        model,
        labels_tokenizer,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        val_per_epoch=args.VAL_PER_EPOCH,
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        logger=logger,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        ckpt_save_mode=cfg.MODEL.SAVE_MODE,
        max_ckpt_save_num=args.max_ckpt_save_num,
        dist_train=dist_train,
        max_num = args.num_token

    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info(f'********************** End training {cfg.EXP_GROUP_PATH}/{cfg.TAG}({args.extra_tag}) **********************\n\n\n')


if __name__ == '__main__':
    main()
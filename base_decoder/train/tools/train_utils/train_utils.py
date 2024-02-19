# This file is modified from https://github.com/traveller59/second.pytorch
# Training function for image-to-text task
# Written by Bo Zhang, Renqiu Xia, Haoyang Peng
# All Rights Reserved 2024-2025.

import os
import torch
import shutil

import tqdm
import time
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from p2s.utils import common_utils, commu_utils
from tools.eval_utils.eval_utils import eval_one_epoch


def loss_function(logits: Tensor, target: Tensor, nvocabs=50244) -> Tensor:
    loss_config ={'reduction': 'none','ignore_index': 0}
    loss_per_word = F.cross_entropy(
        logits.reshape(-1, nvocabs),
        target.reshape(-1),
        **loss_config)
    loss_per_word = loss_per_word.reshape(target.shape)
    final_loss = torch.sum(loss_per_word * (target != 0).float()) / torch.sum(
        torch.sum(target != 0).float() + 1e-6
        )
    return final_loss

def train_one_epoch(model, optimizer, train_loader, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, relaxed_accuracy=0.):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
        
        net_device = next(model.parameters()).device
        # Option One: Tokenizer class into the collator, which causes the data_time slow
        # labels = batch.pop("labels").to(net_device)
        # inputs = batch.pop("inputs").to(net_device)
        # ----------------------------------------------------------
        # Option Two: Preprocess the raw data using the Tokenizer offline
        labels = batch.pop("labels").to(net_device)
        inputs = {
            "flattened_patches": batch.pop("flattened_patches").to(net_device),
            "attention_mask": batch.pop("attention_mask").to(net_device)
            }

        model.train()
        # commu_utils.synchronize()
        optimizer.zero_grad()

        logits = model(**inputs, labels=labels).logits
        loss = loss_function(logits, labels)

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1

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
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            }

            pbar.update()
            # pbar.set_postfix(dict(total_it=accumulated_iter))
            pbar.set_postfix(disp_dict)
            tbar.set_postfix(dict(total_it=accumulated_iter, relaxed_accuracy=relaxed_accuracy))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
        
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, tokenizer, optimizer, criterion, train_loader, val_loader, val_per_epoch, lr_scheduler, optim_cfg, start_epoch, 
                total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ckpt_save_mode, logger,
                train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, dist_train=False, max_num = 200):
    accumulated_iter = start_iter
    best_acc = 0.
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            logger.info('*************** Start Training EPOCH %s *****************' % cur_epoch)
            
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, relaxed_accuracy=best_acc
            )
            logger.info('*************** Start Evaluation EPOCH %s *****************' % cur_epoch)

            if val_per_epoch:
                val_acc = eval_one_epoch(
                    model, tokenizer, val_loader, criterion,
                    rank=rank, dist_test=dist_train, max_num=max_num)
                
                logger.info('*************** Performance of EPOCH %s *****************' % cur_epoch)
                logger.info('*************** Performance = %s *****************' % val_acc)
                
                trained_epoch = cur_epoch + 1
                if tb_log is not None:
                    tb_log.add_scalar('val/acc', val_acc, trained_epoch)
                    
                is_best = val_acc > best_acc
                best_acc = max(best_acc, val_acc)
                
                # save trained model
                if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                    
                    if ckpt_save_mode == "huggingface":
                        ckpt_save_dir_last = str(ckpt_save_dir) + "_last"
                        os.makedirs(ckpt_save_dir_last, exist_ok=True)
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            model.module.save_pretrained(ckpt_save_dir_last)
                        else:
                            model.save_pretrained(ckpt_save_dir_last)
                        
                        state_dict = {
                            'epoch': cur_epoch,
                            'optimizer': optimizer.state_dict(),
                            'best_acc': best_acc,
                            "it": accumulated_iter
                        }
                        torch.save(state_dict, f"{ckpt_save_dir_last}/state_dict.pth")
                        
                        if is_best:
                            ckpt_save_dir_best = str(ckpt_save_dir) + "_best"
                            if os.path.exists(ckpt_save_dir_best):
                                shutil.rmtree(ckpt_save_dir_best)
                            shutil.copytree(f"{ckpt_save_dir_last}", f"{ckpt_save_dir_best}")
                            # shutil.copyfile(f"{ckpt_save_dir_last}/state_dict.pth", f"{ckpt_save_dir_best}/state_dict.pth")
                    
                    elif ckpt_save_mode == "pytorch":
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            # model_state = model_state_to_cpu(model.module.state_dict())
                            model_state = model.module.state_dict()
                            
                        else:
                            model_state = model.state_dict()
                        save_checkpoint(
                        {
                            'epoch': cur_epoch,
                            'state_dict': model_state,
                            'optimizer': optimizer.state_dict(),
                            'best_acc': best_acc,
                            "it": accumulated_iter
                        }, is_best, os.path.join(ckpt_save_dir)
                    )
                    
                    else:
                        raise NotImplementedError
            else:
                # save trained model
                trained_epoch = cur_epoch + 1
                if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                    
                    if ckpt_save_mode == "huggingface":
                        ckpt_save_dir_last = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                        os.makedirs(ckpt_save_dir_last, exist_ok=True)
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            model.module.save_pretrained(ckpt_save_dir_last)
                        else:
                            model.save_pretrained(ckpt_save_dir_last)
                        
                        state_dict = {
                            'epoch': cur_epoch,
                            'optimizer': optimizer.state_dict(),
                            "it": accumulated_iter
                        }
                        torch.save(state_dict, f"{ckpt_save_dir_last}/state_dict.pth")
                    else:
                        raise NotImplementedError
            logger.info('*************** Checkpoint saved! *****************')
            
            
def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


# def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
#     optim_state = optimizer.state_dict() if optimizer is not None else None
#     if model is not None:
#         if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#             model_state = model_state_to_cpu(model.module.state_dict())
#         else:
#             model_state = model.state_dict()
#     else:
#         model_state = None

#     try:
#         import p2s
#         version = 'p2s+' + p2s.__version__
#     except:
#         version = 'none'

#     return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


# def save_checkpoint(state, filename='checkpoint'):
#     if False and 'optimizer_state' in state:
#         optimizer_state = state['optimizer_state']
#         state.pop('optimizer_state', None)
#         optimizer_filename = '{}_optim.pth'.format(filename)
#         torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

#     filename = '{}.pth'.format(filename)
#     torch.save(state, filename)


def save_checkpoint(state, is_best, sav_path, filename='model_last.pth'):
    epoch = state['epoch']
    filename = os.path.join(sav_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(sav_path, 'model_best.pth'))

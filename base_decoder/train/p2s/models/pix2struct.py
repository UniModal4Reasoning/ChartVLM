# Implementation of Pix2Struct Model
# Reference https://arxiv.org/abs/2210.03347 and https://arxiv.org/abs/2309.11268
# Written by Bo Zhang, Renqiu Xia, Haoyang Peng
# All Rights Reserved 2024-2025.

import os
import torch
import torch.nn as nn
from transformers import Pix2StructForConditionalGeneration


class Pix2struct(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.model_name = args.MODEL.MODEL_NAME
        
        # Load the local model checkpoints
        if args.MODEL.PRETRAIN_MODEL_PATH:
            self.model = Pix2StructForConditionalGeneration.from_pretrained(args.MODEL.PRETRAIN_MODEL_PATH)
        else:
        # Load the online model checkpoints
            self.model = Pix2StructForConditionalGeneration.from_pretrained(args.MODEL.MODEL_NAME)
                
    def forward(self, flattened_patches, attention_mask, labels):
        loss = self.model(
                    flattened_patches=flattened_patches,
                    attention_mask=attention_mask,
                    labels=labels
                )
        
        return loss
    
    def generate(self, flattened_patches, attention_mask):
        results = self.model.generate(flattened_patches, attention_mask)
        return results

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1
        
    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch



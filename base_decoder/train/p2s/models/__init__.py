import torch
import numpy as np
from .pix2struct import Pix2struct
from transformers import AutoTokenizer
from collections import namedtuple


def creat_model_tokenizer(args):
    model = Pix2struct(args)
    
    if args.MODEL.PRETRAIN_MODEL_PATH:
        tokenizer = AutoTokenizer.from_pretrained(args.MODEL.PRETRAIN_MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.MODEL.MODEL_NAME)
        
    return model, tokenizer


# def load_data_to_gpu(batch_dict):
#     for key, val in batch_dict.items():
#         if not isinstance(val, np.ndarray):
#             continue
#         elif key in ['frame_id', 'metadata', 'calib']:
#             continue
#         elif key in ['images']:
#             batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
#         elif key in ['image_shape']:
#             batch_dict[key] = torch.from_numpy(val).int().cuda()
#         elif key in ['db_flag']:
#             continue
#         else:
#             batch_dict[key] = torch.from_numpy(val).float().cuda()

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.cuda()
        
            

def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict, **forward_args):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict, **forward_args)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
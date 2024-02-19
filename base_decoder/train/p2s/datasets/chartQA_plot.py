# ChartQADeplot Dataloader for Chart Data
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Bo Zhang, Renqiu Xia, Haoyang Peng
# All Rights Reserved 2024-2025.

import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset


class ChartQADeplot(Dataset):
    def __init__(
            self,
            root,
            split: str = "train",
            subset: str = "Deplot",
            max_patches: int = 4096,  # Base: 4096  Large: 3072
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.max_patches = max_patches
        
        self.input_flattened_patches = sorted(glob(os.path.join(self.root, self.split, "*input_flattened_patches.npy")))
        self.input_attention_mask = sorted(glob(os.path.join(self.root, self.split, "*input_attention_mask.npy")))
        self.label = sorted(glob(os.path.join(self.root, self.split, "*label.npy")))
        

    def __getitem__(self, idx: int):
        input_flattened_patches = torch.from_numpy(np.load(self.input_flattened_patches[idx]))
        input_attention_mask = torch.from_numpy(np.load(self.input_attention_mask[idx]))
        label = torch.from_numpy(np.load(self.label[idx]))
 
        return input_flattened_patches, input_attention_mask, label

    def __len__(self) -> int:
        return len(self.input_flattened_patches)


if __name__ == '__main__':
    dataset = ChartQADeplot()
    item = next(iter(dataset))
    
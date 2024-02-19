# ChartQA Dataloader for Chart Data
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Bo Zhang, Renqiu Xia, Haoyang Peng
# All Rights Reserved 2024-2025.

import torch
import numpy as np

from p2s.datasets.base_dataset import ChartQABASE


class ChartQA(ChartQABASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        input_flattened_patches = torch.from_numpy(np.load(self.input_flattened_patches[idx]))
        input_attention_mask = torch.from_numpy(np.load(self.input_attention_mask[idx]))
        label = torch.from_numpy(np.load(self.label[idx]))
 
        return input_flattened_patches, input_attention_mask, label

    def __len__(self) -> int:
        return len(self.input_flattened_patches)


if __name__ == '__main__':
    dataset = ChartQA()
    item = next(iter(dataset))

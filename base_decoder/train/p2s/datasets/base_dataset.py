# Base Dataloader for Chart Data
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Bo Zhang, Renqiu Xia, Haoyang Peng
# All Rights Reserved 2024-2025.

import os
from glob import glob
from torch.utils.data import Dataset

class ChartQABASE(Dataset):
    def __init__(
            self,
            root: str = "ChartQA_Dataset",
            split: str = "train",
            subset: str = "human" ,  # W: whole    H: human    M: augmented
            max_patches: int = 4096,  # Base: 4096  Large: 3072
    ):
        super().__init__()

        assert split in ["train", "val", "test"]
        assert subset in ["human", "augmented", "merge"]
        
        self.root = root
        self.split = split
        self.subset = subset
        self.max_patches = max_patches

        if subset in ["human", "augmented"]:
            self.input_flattened_patches = sorted(glob(os.path.join(self.root, subset, self.split, "*input_flattened_patches.npy")))
            self.input_attention_mask = sorted(glob(os.path.join(self.root, subset, self.split, "*input_attention_mask.npy")))
            self.label = sorted(glob(os.path.join(self.root, self.subset, subset, "*label.npy")))
        
        else:
            self.input_flattened_patches = []
            self.input_attention_mask = []
            self.label = []
            for item in ["human", "augmented"]:
                self.input_flattened_patches.extend(sorted(glob(os.path.join(self.root, item, self.split, "*input_flattened_patches.npy"))))
                self.input_attention_mask.extend(sorted(glob(os.path.join(self.root, item, self.split, "*input_attention_mask.npy"))))
                self.label.extend(sorted(glob(os.path.join(self.root, item, self.split, "*label.npy"))))
                

        # print('Done')

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses should implement this!")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses should implement this!")

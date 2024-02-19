import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data.dataset import ConcatDataset

from p2s.utils import common_utils

from .chartQA_QA import ChartQA
from .chartQA_plot import ChartQADeplot

DATASET_FUNCTIONS = {
    "QA": ChartQA,
    "Deplot": ChartQADeplot,
}

# COLLATOR_FUNCTION = {
#     "QA": QA_collator,
#     "Deplot": Deplot_collator,
# }

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)



def collator(batch):
    
    new_batch = {
        "flattened_patches": torch.cat([item[0] for item in batch], dim=0), 
        "attention_mask": torch.cat([item[1] for item in batch], dim=0), 
        "labels": torch.cat([item[2] for item in batch], dim=0)
    }

    return new_batch

def build_dataloader(mission_name, data_root, subset, max_patches, batch_size, dist, workers=4, training=True):
    dataset_function = DATASET_FUNCTIONS[mission_name]
    # collator_function = COLLATOR_FUNCTION[args.mission_name]
    
    dataset = dataset_function(
        root=data_root,
        split="train" if training else "val",
        subset=subset,
        max_patches=max_patches,
    )
    
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=collator,
        drop_last=False, sampler=sampler, timeout=0, persistent_workers=True if workers > 1 else False
    )

    return dataset, dataloader, sampler  # , val_loader, val_sampler


def build_dataloader_multi_db(mission_name, data_root_1, data_root_2,
                             subset, max_patches, batch_size, dist, 
                             workers=4, training=True):
    dataset_function = DATASET_FUNCTIONS[mission_name]
    # collator_function = COLLATOR_FUNCTION[args.mission_name]

    dataset_1 = dataset_function(
        root=data_root_1,
        split="train" if training else "val",
        subset=subset,
        max_patches=max_patches,
    )
    dataset_2 = dataset_function(
        root=data_root_2,
        split="train" if training else "val",
        subset=subset,
        max_patches=max_patches,
    )

    concat_dataset = ConcatDataset([dataset_1, dataset_2])
    
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(concat_dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(concat_dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        concat_dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=collator,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset_1, dataset_2, dataloader, sampler
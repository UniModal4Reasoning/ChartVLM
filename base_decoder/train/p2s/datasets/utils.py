import torch
from transformers import AutoTokenizer


def QA_collator(batch, tokenizer: AutoTokenizer):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["labels"] for item in batch]
    text_inputs = tokenizer(
        text=texts,
        return_tensors="pt",
        # add_special_tokens=True,
        # max_length=50,
        # padding="max_length",
        padding=True,
        # truncation=True
    )

    new_batch["labels"] = text_inputs["input_ids"]

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch


def Deplot_collator(batch, tokenizer: AutoTokenizer):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["labels"] for item in batch]
    text_inputs = tokenizer(
        text=texts,
        return_tensors="pt",
        # add_special_tokens=True,
        # max_length=50,
        # padding="max_length",
        padding=True,
        # truncation=True
    )

    new_batch["labels"] = text_inputs["input_ids"]

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch
from typing import Union, Callable
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.utils import prepare_dolly

PAD_TOKEN_ID = 0
SEED = 42


class CustomDataset(Dataset):
    def __init__(self, data: dict, tokenizer, collate_fn: Callable = None):
        # Process the data
        if collate_fn is not None:
            self.data = collate_fn(data, tokenizer)
        else:
            self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text from the data
        input_ids = self.data[idx]["input_ids"]
        input_ids_no_response = self.data[idx]["input_ids_no_response"]
        labels = self.data[idx]["labels"]
        return {"input_ids": input_ids, "input_ids_no_response": input_ids_no_response, "labels": labels}


def create_dataloaders(data: pd.DataFrame, tokenizer, batch_size=8, shuffle=True, num_workers=4):
    # Load the dataset

    dataset = CustomDataset(data, tokenizer)
    # Create the dataloaders
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)

    # Return the dataloaders
    return dataloader


def data_loading_custom_dataset(
    dataset_path: str,
    tokenizer_name: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
):
    # Load the dataset
    if ".csv" in dataset_path:
        data = pd.read_csv(dataset_path).to_dict("dict")
    elif ".json" in dataset_path:
        data = pd.read_json(dataset_path).to_dict("dict")
    elif ".xlsx" in dataset_path:
        data = pd.read_excel(dataset_path).to_dict("dict")
    else:
        raise "Couldn't find the data in path"
    # Create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer()
    dataset = CustomDataset(data, tokenizer=tokenizer, collate_fn=prepare_dolly)
    dataloader = create_dataloaders(dataset, tokenizer, batch_size, shuffle, num_workers)
    return dataloader


def data_loading_hf_dataset(
    dataset_path: str,
    tokenizer_name: str,
    test_size: int,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
):
    # Load the dataset

    data = load_dataset(dataset_path)
    # Create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    train_size = len(data) - test_size
    train_set, test_set = random_split(
        data,
        lengths=(train_size, test_size),
        generator=torch.Generator().manual_seed(SEED),
    )
    train_dataset = CustomDataset(train_set, tokenizer=tokenizer, collate_fn=prepare_dolly)
    test_dataset = CustomDataset(test_set, tokenizer=tokenizer, collate_fn=prepare_dolly)
    train_dataloader = create_dataloaders(train_dataset, tokenizer, batch_size, shuffle, num_workers)
    test_dataloader = create_dataloaders(test_dataset, tokenizer, batch_size, shuffle, num_workers)
    return train_dataloader, test_dataloader
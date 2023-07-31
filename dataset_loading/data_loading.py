from typing import Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.utils import process_llm_data


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, collate_fn: function = None):
        # Process the data
        if collate_fn is not None:
            self.data = collate_fn(data)
        else:
            self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text from the data
        text = self.data.loc[idx, "text"]
        # Tokenize the text
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        # Return the encoding
        return encoding


def create_dataloaders(data: pd.DataFrame, tokenizer, batch_size=8, shuffle=True, num_workers=4):
    # Load the dataset

    dataset = CustomDataset(data, tokenizer)
    # Create the dataloaders
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers)

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
        data = pd.read_csv(dataset_path)
    elif ".json" in dataset_path:
        data = pd.read_json(dataset_path)
    elif ".xlsx" in dataset_path:
        data = pd.read_excel(dataset_path)
    else:
        raise "Couldn't find the data in path"
    # Create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataloader = create_dataloaders(data, tokenizer, batch_size, shuffle, num_workers)
    return dataloader


def data_loading_hf_dataset(
    dataset_path: str,
    tokenizer_name: str,
    split_name: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
):
    # Load the dataset

    data = load_dataset(dataset_path)
    # Create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return create_dataloaders(pd.DataFrame(data[split_name]), tokenizer, batch_size, shuffle, num_workers)


data_loading_hf_dataset("databricks/databricks-dolly-15k", "gpt2", "train")

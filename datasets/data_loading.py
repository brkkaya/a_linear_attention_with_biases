from typing import Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class HFDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = self.process_llm_data(data)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def process_llm_data(self, data: pd.DataFrame):
        """data_sample consists of three keys, prompt, input, output, concat them"""
        text_col = []
        for instance in data:
            prompt = instance["prompt"]
            input = instance["input"]
            output = instance["output"]
            if len(input.strip()) == 0:
                text = "### Instruction: \n" + prompt + "###Response: \n" + output
            else:
                text = "### Instruction: \n" + prompt + "\n### Input: \n" + input + "###Response: \n" + output

            text_col.append(text)

            data.loc[:, "text"] = text_col
        return data

    def __getitem__(self, idx):
        text = self.data.loc[idx, "text"]
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        return encoding


def create_dataloaders(data, tokenizer, model_name, batch_size=8, shuffle=True, num_workers=4):
    # Load the dataset

    train_dataset = HFDataset(data["train"], tokenizer, model_name)
    test_dataset = HFDataset(data["test"], tokenizer, model_name)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle, num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size, False, num_workers)

    return train_dataloader, test_dataloader

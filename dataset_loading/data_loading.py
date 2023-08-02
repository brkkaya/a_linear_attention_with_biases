from typing import Union, Callable
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.utils import generate_prompt

tokenizer = AutoTokenizer.from_pretrained("gpt2")
SEED = 42


def import_data(data_name_or_path: str):
    if "json" in data_name_or_path:
        data = load_dataset("json", data_files=data_name_or_path)
    elif "csv" in data_name_or_path:
        data = load_dataset("csv", data_files=data_name_or_path)
    else:
        data = load_dataset(data_name_or_path)
    return data


def tokenize(prompt: str, add_oes_token: bool = True):
    result = tokenizer(prompt, return_tensors=None, padding=False)
    if result["input_ids"][-1] != tokenizer.eos_token_id and add_oes_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def process_tokenize_instance(instance: dict):
    text_input = generate_prompt(instance)
    tokenized_input = tokenize(prompt=text_input, add_oes_token=True)
    return tokenized_input


def apply_process(data_name_or_path: str):
    data = import_data(data_name_or_path)
    data["train"].train_test_split(test_size=200, seed=SEED, shuffle=True)
    train_data = data["train"].map(process_tokenize_instance)
    test_data = data["test"].map(process_tokenize_instance)

def data_collator():
    # re-implementation of transformers.DataCollatorSeq2Seq
    pass
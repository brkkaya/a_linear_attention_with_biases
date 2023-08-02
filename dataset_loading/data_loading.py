from typing import Union, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset
from utils.utils import generate_prompt

tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = 0
tokenizer.special_tokens_map
SEED = 42


def import_data(data_name_or_path: str):
    if "json" in data_name_or_path:
        data = load_dataset("json", data_files=data_name_or_path)
    elif "csv" in data_name_or_path:
        data = load_dataset("csv", data_files=data_name_or_path)
    else:
        data = load_dataset(data_name_or_path)
    return data


def tokenize(prompt: str, label_pad_token_id: int = -100, add_oes_token: bool = True):
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


def apply_padding(features: Dataset,max_length:int = 1024, padding_size: str = "left", label_pad_token_id: int = -100):
    labels = [feature["labels"] for feature in features]
    max_label_length = max(len(l) for l in labels)

    tokenizer.padding_side = padding_size
    for feature in features:
        remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
        feature["labels"] = (
            feature["labels"] + remainder if tokenizer.padding_side == "right" else remainder + feature["labels"]
        )
    #TODO without text field it worked but not add path to fields
    features = features.map(lambda e: tokenizer.pad(e,max_length=max_length, padding=True, return_tensors="pt"),batched=True)
    return features


def apply_process(data_name_or_path: str):
    data = import_data(data_name_or_path)
    data = data["train"].train_test_split(test_size=200, seed=SEED, shuffle=True)
    # train_data = data["train"].map(process_tokenize_instance)
    test_data = data["test"].map(process_tokenize_instance)
    apply_padding(test_data)
    return train_data, test_data


def data_collator(features: dict, tokenizer: PreTrainedTokenizerBase, label_pad_token_id: int = -100):
    # re-implementation of transformers.DataCollatorSeq2Seq
    labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
    if labels is not None:
        max_label_length = max(len(l) for l in labels)

    padding_side = tokenizer.padding_side
    for feature in features:
        remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
        feature["labels"] = feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
    features = tokenizer.pad(features, padding=True, return_tensors=True)

    return features


apply_process("data/alpaca_cleaned.json")

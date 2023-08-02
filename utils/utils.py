import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'output' field."""

    if example["input"]:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}\n"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}\n"


def import_data(data_name_or_path:str):
    if "json" in data_name_or_path:
        data = load_dataset("json",data_files=data_name_or_path)
    elif "csv" in data_name_or_path:
        data = load_dataset("csv",data_files=data_name_or_path)
    else:
        data = load_dataset(data_name_or_path)
    print()
import_data("data/alpaca_cleaned.json")
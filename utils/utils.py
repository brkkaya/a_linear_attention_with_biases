import torch
from transformers import AutoTokenizer
from tqdm import tqdm


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'output' field."""

    if example["input"]:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}\n"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}\n"

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
IGNORE_INDEX = -1


def prepare_dataset(data: list, tokenizer: AutoTokenizer, max_length: int = None):
    if max_length is None:
        max_length = tokenizer.model_max_length
    return [prepare_instance(sample, tokenizer=tokenizer, max_length=max_length) for sample in tqdm(data)]


def prepare_instance(example: dict, tokenizer: AutoTokenizer, max_length: int, mask_inputs: bool = True):
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt, encoded_full_prompt_len = tokenize(
        full_prompt, tokenizer=tokenizer, max_length=max_length, use_bos=True, use_eos=False
    )
    encoded_full_prompt_and_response, _ = tokenize(
        full_prompt_and_response, tokenizer=tokenizer, use_bos=True, use_eos=True, max_length=max_length
    )

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: encoded_full_prompt_len] = IGNORE_INDEX

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def tokenize(
    text: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    use_bos: bool = True,
    use_eos: bool = True,
    use_pad: bool = True,
):
    sample = tokenizer(
        text,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
        return_special_tokens_mask=False,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]

    if use_eos:
        sample = sample + [tokenizer.eos_token_id]
    if use_bos:
        sample = [tokenizer.bos_token_id] + sample

    sample_length = len(sample)
    padding_length = max_length - sample_length

    if use_pad and max_length > sample_length:
        sample += [tokenizer.pad_token_id] * padding_length
        sample = torch.tensor(sample, dtype=torch.int)
        return sample, sample_length
    else:
        sample = torch.tensor(sample, dtype=torch.int)
        return sample, sample_length


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:"

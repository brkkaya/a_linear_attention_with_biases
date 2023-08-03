from torch.utils.data import Dataset
from transformers import  PreTrainedTokenizerBase
from datasets import load_dataset
from utils.utils import generate_prompt
SEED = 42

class DataLoading:
    """Returns a transformers Dataset, it is compatible with torch.utils.data.Dataset, you can use it directly with 
    torch.utils.data.DataLoader. But must use apply_padding when to dataloader, """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, seed: int = 42) -> None:
        self.tokenizer = tokenizer
        self.seed = seed

    def import_data(self, data_name_or_path: str):
        if "json" in data_name_or_path:
            data = load_dataset("json", data_files=data_name_or_path)
        elif "csv" in data_name_or_path:
            data = load_dataset("csv", data_files=data_name_or_path)
        else:
            data = load_dataset(data_name_or_path)
        return data

    def tokenize(self, prompt: str, add_oes_token: bool = True):
        result = self.tokenizer(prompt, return_tensors=None, padding=False)
        if result["input_ids"][-1] != self.tokenizer.eos_token_id and add_oes_token:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def process_tokenize_instance(self, instance: dict):
        text_input = generate_prompt(instance)
        tokenized_input = self.tokenize(prompt=text_input, add_oes_token=True)

        return tokenized_input

    def apply_padding(self, features: Dataset, padding_size: str = "left", label_pad_token_id: int = -100):
        labels = [feature["labels"] for feature in features]
        max_label_length = max(len(l) for l in labels)

        self.tokenizer.padding_side = padding_size
        for idx, feature in enumerate(features):
            remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
            feature["labels"] = (
                feature["labels"] + remainder
                if self.tokenizer.padding_side == "right"
                else remainder + feature["labels"]
            )
            features[idx] = feature

        features = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        return features

    def apply_process(self, data_name_or_path: str):
        data = self.import_data(data_name_or_path)
        data = data["train"].train_test_split(test_size=200, seed=self.seed, shuffle=True)
        train_data = data["train"].map(self.process_tokenize_instance)
        train_data = train_data.remove_columns(["instruction", "input", "output"])

        test_data = data["test"].map(self.process_tokenize_instance)
        test_data = test_data.remove_columns(["instruction", "input", "output"])
        return train_data, test_data

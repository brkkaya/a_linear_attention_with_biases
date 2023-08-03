import torch
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, AutoTokenizer
from dataset_loading.data_loading import DataLoading
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the model and other necessary objects
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)  # Define your model
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Define your tokenizer
tokenizer.pad_token_id = 0
data_process = DataLoading(tokenizer=tokenizer)
train_data, test_data = data_process.apply_process("data/alpaca_cleaned.json")

BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 2e-4
OUTPUT_DIR = "experiments"

training_arguments = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    learning_rate=LEARNING_RATE,
    # fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard",
    no_cuda=True,
)
trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    args=training_arguments,
    data_collator=data_process.apply_padding,
)

# model = torch.compile(model)
trainer.train()
model.save_pretrained(OUTPUT_DIR)

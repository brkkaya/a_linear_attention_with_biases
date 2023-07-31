from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, AutoTokenizer
from dataset_loading.data_loading import data_loading_hf_dataset

# Define the model and other necessary objects
model = GPT2LMHeadModel.from_pretrained("gpt2")  # Define your model
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Define your tokenizer
tokenizer.add_special_tokens({"pad_token": "<pad>"})
model.resize_token_embeddings(len(tokenizer))
train_dataset, test_dataset = data_loading_hf_dataset("databricks/databricks-dolly-15k", "gpt2", test_size=2000)
# train_dataloader = create_dataloaders(train_dataset, tokenizer, batch_size, shuffle, num_workers)
# test_dataloader = create_dataloaders(test_dataset, tokenizer, batch_size, shuffle, num_workers)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./saved_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    # logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=5,
    save_steps=10000,
    bf16=True,no_cuda=True)

# Define the trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

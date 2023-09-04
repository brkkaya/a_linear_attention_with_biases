import os
from transformers import GPT2LMHeadModel, AutoTokenizer, get_linear_schedule_with_warmup
from dataset_loading.data_loading import DataLoading
from torch.utils.data import DataLoader
from tqdm import tqdm
import accelerate
import torch
from models.transformer import Decoder


DEVICE = "cuda"
BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
NUM_EPOCHS = 2
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 2e-4
OUTPUT_DIR = "experiments"
WARMUP_STEPS = 300

# Define the model and other necessary objects
accelerator = accelerate.Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Define your tokenizer
model =  Decoder(48,n_head=2,n_layer=2,vocab_dim=tokenizer.vocab_size).to(DEVICE) # Define your model
tokenizer.pad_token_id = 0
data_process = DataLoading(tokenizer=tokenizer)
train_data, test_data = data_process.apply_process("data/alpaca_cleaned.json")


train_loader = DataLoader(train_data, batch_size=MICRO_BATCH_SIZE, collate_fn=data_process.apply_padding, shuffle=True)
test_loader = DataLoader(test_data, batch_size=MICRO_BATCH_SIZE, collate_fn=data_process.apply_padding, shuffle=False)

# Prepare optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
test_loader = accelerator.prepare_data_loader(test_loader)
# Training loop
model.zero_grad()
for epoch in range(NUM_EPOCHS):
    model.train()
    for step, batch in tqdm(enumerate(train_loader)):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        with accelerator.accumulate(model):
            outputs = model(inputs, mask=attention_mask, targets=labels)
            loss = outputs.loss
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if step % 50 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Step [{step}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    # Save the model at the end of each epoch
    output_epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
    os.makedirs(output_epoch_dir, exist_ok=True)
    model.save_pretrained(output_epoch_dir)

    # Evaluation on the test set at the end of each epoch
    model.eval()
    total_test_loss = 0
    num_test_batches = 0
    with torch.no_grad():
        for test_batch in test_loader:
            inputs = test_batch["input_ids"]
            labels = test_batch["labels"]
            attention_mask = test_batch["attention_mask"]

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            test_loss = outputs.loss
            total_test_loss += test_loss.item()
            num_test_batches += 1

    average_test_loss = total_test_loss / num_test_batches
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Average Test Loss: {average_test_loss:.4f}")

print("Training complete!")

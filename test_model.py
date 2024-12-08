import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    AdamW
)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Train a Custom Tokenizer
from tokenizers import ByteLevelBPETokenizer

# Path to your dataset (plain text file)
dataset_path = "C:/Users/User/Desktop/research/ani.txt"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The file '{dataset_path}' does not exist.")

# Train tokenizer if not already trained
tokenizer_path = "C:/Users/User/Desktop/research/tokenizer"
os.makedirs(tokenizer_path, exist_ok=True)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[dataset_path], vocab_size=50257, min_frequency=2)

# Save tokenizer
tokenizer.save_model(tokenizer_path)
print("Tokenizer trained and saved.")

# Step 2: Load tokenizer with Hugging Face
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Define a Text Dataset
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=512):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=True)
        self.examples = [
            torch.tensor(tokens[i: i + block_size], dtype=torch.long)
            for i in range(0, len(tokens) - block_size, block_size)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Initialize dataset and dataloader
block_size = 128
batch_size = 8
dataset = TextDataset(dataset_path, tokenizer, block_size=block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 4: Define the Transformer Model
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=block_size,
    n_embd=256,  # Reduced embedding size
    n_layer=4,   # Reduced number of layers
    n_head=4     # Reduced number of attention heads
)
model = GPT2LMHeadModel(config).to(device)

# Step 5: Train the Model
optimizer = AdamW(model.parameters(), lr=5e-4)
epochs = 500
model.train()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

print("Training complete.")

# Step 6: Save the Model and Tokenizer
model_save_path = "C:/Users/User/Desktop/research/custom_model"
os.makedirs(model_save_path, exist_ok=True)

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved at '{model_save_path}'.")

# Step 7: Load and Use the Saved Model
# Reload model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_save_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# Text generation example
model.eval()

# Function to generate text
def generate_text(prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompt based on training data
prompt = "intishar"  # Replace with a prompt relevant to your data
generated_text = generate_text(prompt, max_length=100)
print(f"Generated Text: {generated_text}")

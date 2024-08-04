#!/usr/bin/env python
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

def train(model, train_dataloader, val_dataloader, epochs=3, lr=2e-5, mode="mlm", 
          patience=3, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    if mode == "mlm":
        criterion = nn.CrossEntropyLoss()
    elif mode == "contrastive":
        criterion = nn.TripletMarginLoss()
    else:
        raise ValueError("Mode must be 'mlm' or 'contrastive'")

    # Early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        #for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            if mode == "mlm":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            elif mode == "contrastive":
                anchor = {k: v.to(device) for k, v in batch["anchor"].items()}
                positive = {k: v.to(device) for k, v in batch["positive"].items()}
                negative = {k: v.to(device) for k, v in batch["negative"].items()}

                anchor_emb = model(**anchor)
                positive_emb = model(**positive)
                negative_emb = model(**negative)

                loss = criterion(anchor_emb, positive_emb, negative_emb)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            print(f"\rEpoch {epoch+1}/{epochs}: step {step+1:3d}/{len(train_dataloader)}\t\tLoss: {loss.item():.4f}", end="")
        print()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}: Average train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_eval_loss = 0

        with torch.no_grad():
            #for batch in tqdm(val_dataloader, desc="Validation"):
            for step, batch in enumerate(val_dataloader):
                if mode == "mlm":
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                elif mode == "contrastive":
                    anchor = {k: v.to(device) for k, v in batch["anchor"].items()}
                    positive = {k: v.to(device) for k, v in batch["positive"].items()}
                    negative = {k: v.to(device) for k, v in batch["negative"].items()}

                    anchor_emb = model(**anchor)
                    positive_emb = model(**positive)
                    negative_emb = model(**negative)

                    loss = criterion(anchor_emb, positive_emb, negative_emb)

                total_eval_loss += loss.item()

                print(f"\rEpoch {epoch+1}/{epochs}: Val step {step+1:3d}/{len(train_dataloader)}\t\tLoss: {loss.item():.4f}", end="")
            print()

        avg_val_loss = total_eval_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}: Average validation loss: {avg_val_loss:.4f}")

        # Checkpoints
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_ep_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, checkpoint_path)
            print(f"New best model saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model
    best_model_path = os.path.join(checkpoint_dir, f"best_model_ep_{epoch+1-epochs_without_improvement}.pt")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from {best_model_path}")
 
    return model

## Usage example:
#model_name = "bert-base-uncased"
#mode = "mlm"  # or "contrastive"
#
#model = FTModel(model_name, mode)
#train_dataloader = get_dataloader(tokenizer, train_texts, mode)
#val_dataloader = get_dataloader(tokenizer, val_texts, mode, shuffle=False)
#
#trained_model = train(model, train_dataloader, val_dataloader, epochs=3, lr=2e-5, mode=mode, patience=5)


""" old
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
mode = "mlm"  # or "contrastive"
model = FTModel("bert-base-uncased", mode)

if mode == "mlm":
    train_dataset = MLMDataset(train_texts, tokenizer)
    val_dataset = MLMDataset(val_texts, tokenizer)
else:
    train_dataset = ContrastiveDataset(train_texts, tokenizer)
    val_dataset = ContrastiveDataset(val_texts, tokenizer)

train_dataloader = get_dataloader(train_dataset)
val_dataloader = get_dataloader(val_dataset)

trained_model = train(model, train_dataloader, val_dataloader, epochs=10, lr=2e-5, mode=mode, patience=3)
"""
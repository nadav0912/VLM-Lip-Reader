import os
import sys
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports from our modules ---
from modeling.loaders.singleWordDataset import SingleWordDataset
from modeling.loaders.vocab import Vocabulary
from utils.torch_dataset import pad_collate, print_dataset_stats
from modeling.architectures.cnn_encoder import CNNEncoder

# --- Config ---
load_dotenv()
RUN_NAME = "PRETRAIN_CNN_VOCAB_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.getenv("RUN_DIR", "runs")
CHECKPOINT_DIR = os.path.join(RUN_DIR, RUN_NAME)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")
METRICS_PATH = os.path.join(CHECKPOINT_DIR, "metrics.json")
BEST_PREDS_PATH = os.path.join(CHECKPOINT_DIR, "best_preds.json")

# HYPERPARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Set the model to training mode (applies Dropout, etc.)
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Train", leave=False)

    for videos, labels, lengths in loop:
        # 1. Transfer data to GPU
        videos = videos.to(device) # (B, 1, T, 88, 88)
        labels = labels.to(device) # (B,)
        
        # 2. Reset gradients
        optimizer.zero_grad()
        
        # 3. Forward Pass
        outputs = model(videos) # (B, num_classes)
        
        # 4. Calculate loss
        loss = criterion(outputs, labels)
        
        # 5. Backward Pass (update weights)
        loss.backward()
        optimizer.step()
        
        # 6. Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1) # Which word has the highest score?
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update the progress bar
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval() # Validation mode (disables Dropout)
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    
    with torch.no_grad(): # Don't calculate gradients (saves memory)
        for videos, labels, lengths in tqdm(loader, desc="Val", leave=False):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_preds, all_labels

def main():
    print(f"🚀 Starting Training on {DEVICE}...")
    
    labels_path = os.path.join(os.getenv("SINGLE_WORD_CLIPS_DIR"), "labels.json")
    vocab = Vocabulary.from_json_file(labels_path)
    # Save the vocabulary (conversion from word to number) to a JSON file
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab.word2idx, f, indent=4, ensure_ascii=False)

    print_dataset_stats(labels_path)

    # 1. Dataset Setup
    print("📂 Loading Dataset...")
    full_dataset = SingleWordDataset(
        video_dir=os.getenv("SINGLE_WORD_CLIPS_DIR"),
        labels_path=os.path.join(os.getenv("SINGLE_WORD_CLIPS_DIR"), "labels.json"),
        vocab=vocab,
        grayscale=True
    )
    
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size} words")
    
    # Split into Train / Validation (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate, num_workers=0)
    
    # 2. Model Setup
    print("🧠 Initializing Model...")
    model = CNNEncoder(num_classes=vocab_size, pretrain_mode=True)
    model = model.to(DEVICE)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Scheduler: Reduce learning rate if there's no improvement for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # 3. Training Loop
    best_acc = 0.0
    best_epoch = 0
    history = [{"epoch": 0, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "train_loss": 0.0, "train_acc": 0.0, "val_loss": 0.0, "val_acc": 0.0, "lr": LEARNING_RATE, "best_epoch": 0}]
    
    for epoch in range(EPOCHS):
        print(f"\nExample Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        
        # Scheduler Step
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Done: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"      Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            save_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)

            preds_data = {
                "epoch": epoch + 1,
                "predictions": val_preds, 
                "targets": val_labels
            }
            with open(BEST_PREDS_PATH, "w") as f:
                json.dump(preds_data, f)
            print(f"🏆 New Best Model Saved! ({best_acc:.2f}%)")

        # Save Metrics
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "best_epoch": best_epoch
        }
        history.append(epoch_stats)
        
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

    print("\n✅ Training Complete.")

if __name__ == "__main__":
    main()
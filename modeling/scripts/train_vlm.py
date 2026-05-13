import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv 
from huggingface_hub import login

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
else:
    print("[!] Warning: HUGGINGFACE_TOKEN not found in .env")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
from modeling.loaders.clips_dataset import LipReadingVLMDataset 
from modeling.architectures.cnn_encoder import CNNEncoder 
from modeling.architectures.LipVLM import LipVLM


# ================= CONFIGURATION =================
STAGE_2_UNFREEZE = False 

VIDEO_DIR = os.getenv("FINAL_VIDEOS_DIR")
JSON_DIR = os.getenv("FINAL_LABELS_DIR")
run_dir = os.getenv("RUN_DIR")

if run_dir:
    CNN_CHECKPOINT = os.path.join(run_dir, "PRETRAIN_CNN_VOCAB_20260215_124233", "best_model.pth")
else:
    CNN_CHECKPOINT = None 

if STAGE_2_UNFREEZE:
    CHECKPOINT_DIR = os.path.join(run_dir, f"FINETUNE_VLM_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
else:
    CHECKPOINT_DIR = os.path.join(run_dir, f"PRETRAIN_VLM_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LOG_FILE = os.path.join(CHECKPOINT_DIR, "training_log.json")

# LLM_ID = "meta-llama/Llama-3.2-1B"
LLM_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Hyperparameters
NUM_EPOCHS = 6 if not STAGE_2_UNFREEZE else 3
BATCH_SIZE = 16 if not STAGE_2_UNFREEZE else 1
GRAD_ACCUMULATION = 2 if not STAGE_2_UNFREEZE else 8
VALIDATION_SPLIT = 0.05  # 5% for validation, 95% for training

# Learning Rates
LR_CNN = 1e-5     
LR_ADAPTER = 1e-4 
LR_LLM = 5e-6     

# ================= LOGIC =================

def save_logs(history):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    print(f"    📊 Log saved to {LOG_FILE}")

def save_checkpoint(model, epoch, val_loss, is_best=False):
    """Saves checkpoint. Creates a special copy if it's the best model so far."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Regular epoch save
    path = os.path.join(CHECKPOINT_DIR, f"lipvlm_s{2 if STAGE_2_UNFREEZE else 1}_e{epoch+1}.pt")
    torch.save(model.state_dict(), path)
    
    # Best model save
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save(model.state_dict(), best_path)
        print(f"    🏆 New Best Model Saved! (Val Loss: {val_loss:.4f})")
    else:
        print(f"    💾 Saved Epoch Checkpoint: {path}")

def validate(model, dataloader, device, tokenizer):
    """Runs validation loop to calculate loss on unseen data."""
    model.eval()
    total_val_loss = 0
    print("    🔍 Running Validation...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            video = batch["video"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            prompt_len = batch["prompt_len"]
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            # Forward pass only, no backprop
            loss, _ = model(video, input_ids, labels, attention_mask, prompt_len)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss

def debug_generation(model, dataset, device, tokenizer):
    """Llama-generated sanity check."""
    model.eval()
    # Pick a random sample from the VALIDATION dataset preferably, or train if index 0
    sample = dataset[0] 
    video = sample["video"].unsqueeze(0).to(device, dtype=torch.bfloat16)
    
    target_ids = sample["labels"].clone()
    target_ids[target_ids == -100] = tokenizer.pad_token_id
    target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

    prompt_ids = sample["input_ids"][:sample["prompt_len"]]
    input_ids = prompt_ids.unsqueeze(0).to(device)
    prompt_len = torch.tensor([sample["prompt_len"]])

    generated_text = ""
    with torch.no_grad():
        dummy_labels = torch.zeros_like(input_ids) 
        loss, logits = model(video, input_ids, dummy_labels, torch.ones_like(input_ids), prompt_len)
        last_token_logits = logits[0, -1, :]
        predicted_id = torch.argmax(last_token_logits).item()
        generated_text = tokenizer.decode([predicted_id])
        
        print(f"\n[👀 Sanity Check] Target: '{target_text[:30]}...' | Predicted: '{generated_text}'")
    
    model.train()
    return {"target": target_text, "prediction_snippet": generated_text}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    
    # 1. Load Components
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize Full Dataset
    full_dataset = LipReadingVLMDataset(VIDEO_DIR, JSON_DIR, tokenizer)
    
    # --- Split Dataset (Train / Val) ---
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    
    print(f"[*] Splitting Dataset: {train_size} Train samples | {val_size} Validation samples")
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize Models
    cnn_encoder = CNNEncoder(pretrain_mode=False)
    if CNN_CHECKPOINT and os.path.exists(CNN_CHECKPOINT):
        print(f"[*] Loading CNN Weights from {CNN_CHECKPOINT}")
        ckpt = torch.load(CNN_CHECKPOINT)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        cnn_encoder.load_state_dict(state_dict, strict=False)
    
    model = LipVLM(cnn_encoder, llm_id=LLM_ID)
    
    if STAGE_2_UNFREEZE:
        model.freeze_llm(False)
    else:
        model.freeze_llm(True)
        
    model.to(device, dtype=torch.bfloat16)

    # 2. Optimizer Setup
    param_groups = []
    param_groups.append({
        'params': filter(lambda p: p.requires_grad, model.visual_encoder.parameters()),
        'lr': LR_CNN
    })
    
    adapter_params = [p for n, p in model.named_parameters() 
                      if 'visual_projection' in n or 'post_adapter_norm' in n]
    param_groups.append({'params': adapter_params, 'lr': LR_ADAPTER})

    if STAGE_2_UNFREEZE:
        llm_params = [p for n, p in model.named_parameters() if 'llm' in n]
        param_groups.append({'params': llm_params, 'lr': LR_LLM})

    optimizer = optim.AdamW(param_groups)

    # 3. Scheduler
    total_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )

    history = []
    best_val_loss = float('inf') # Track best validation loss

    # 4. Training Loop
    print(f"[*] Starting Training. Steps: {total_steps}")
    
    for epoch in range(NUM_EPOCHS):
        # --- TRAIN LOOP ---
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for step, batch in enumerate(progress_bar):
            video = batch["video"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            prompt_len = batch["prompt_len"]
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            loss, _ = model(video, input_ids, labels, attention_mask, prompt_len)
            loss = loss / GRAD_ACCUMULATION
            loss.backward()
            
            if (step + 1) % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * GRAD_ACCUMULATION
            progress_bar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUMULATION:.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION LOOP ---
        avg_val_loss = validate(model, val_loader, device, tokenizer)
        
        print(f"[-] Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Check if this is the best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # Sanity Check (on Validation Set)
        sample_result = debug_generation(model, val_dataset, device, tokenizer)
        
        # Logging
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "is_best": is_best,
            "lr": optimizer.param_groups[1]['lr'],
            "sanity_check": sample_result
        }
        history.append(epoch_log)
        save_logs(history)
        
        # Save Checkpoint
        save_checkpoint(model, epoch, avg_val_loss, is_best)

if __name__ == "__main__":
    train()
import os
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# --- Imports from our modules ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modeling.architectures.cnn_encoder import CNNEncoder
from modeling.loaders.singleWordDataset import SingleWordDataset

# --- Configuration ---
load_dotenv()
TARGET_RUN_NAME = "PRETRAIN_CNN_VOCAB_20260215_124233" 

# Path Setup
RUN_DIR = os.path.join("runs", TARGET_RUN_NAME)
METRICS_PATH = os.path.join(RUN_DIR, "metrics.json")
PREDS_PATH = os.path.join(RUN_DIR, "best_preds.json")
MODEL_PATH = os.path.join(RUN_DIR, "best_model.pth")
VOCAB_PATH = os.path.join(RUN_DIR, "vocab.json")

# Data Directory (Needed to load a sample video for visualization)
DATA_DIR = os.getenv("SINGLE_WORD_CLIPS_DIR", "data/06_word_clips")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")


def plot_training_history():
    """
    Plots the Loss, Accuracy, and Learning Rate curves from the training history.
    """
    if not os.path.exists(METRICS_PATH):
        print(f"❌ Metrics file not found: {METRICS_PATH}")
        return

    print("📊 Plotting training history...")
    with open(METRICS_PATH, 'r') as f:
        history = json.load(f)
    
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(18, 5))

    # 1. Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', linestyle='--', linewidth=2)
    plt.title('Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Accuracy Curve
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc', color='green', linewidth=2)
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc', linestyle='--', color='lime', linewidth=2)
    plt.title('Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Learning Rate
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['lr'], color='orange', linewidth=2)
    plt.title('Learning Rate Decay', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True, alpha=0.3)
    plt.yscale('log') # Log scale helps see small changes

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(top_k=15):
    """
    Plots a normalized Confusion Matrix for the Top-K most frequent words.
    Filtering prevents the chart from becoming unreadable with 500 classes.
    """
    if not (os.path.exists(PREDS_PATH) and os.path.exists(VOCAB_PATH)):
        print("⚠️ Predictions or Vocab file missing. Skipping CM analysis.")
        return

    print("📊 Generating Confusion Matrix...")
    
    # Load data
    with open(PREDS_PATH, 'r') as f:
        data = json.load(f)
    preds = np.array(data['predictions'])
    targets = np.array(data['targets'])

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}

    # --- Filter Logic: Get Top-K Most Frequent Words in Validation Set ---
    # Count occurrences of each target class
    unique, counts = np.unique(targets, return_counts=True)
    
    # Sort indices by count (descending)
    sorted_indices = np.argsort(-counts)
    top_indices = unique[sorted_indices][:top_k]
    
    # Create a mask to filter the data
    mask = np.isin(targets, top_indices)
    
    filtered_preds = preds[mask]
    filtered_targets = targets[mask]
    
    # Get the names for the axis labels
    labels = [idx2word[i] for i in top_indices]

    # Create Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_indices)
    
    # Normalize (to show percentages instead of raw counts)
    # Adding epsilon to avoid division by zero
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix (Top {top_k} Most Frequent Words)', fontsize=16)
    plt.ylabel('True Word')
    plt.xlabel('Predicted Word')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def visualize_model_internals(num_frames_to_show=8):
    """
    1. Visualizes the learned weights (filters) of the first 3D Conv layer.
    2. Visualizes the activation feature maps of the LAST Conv layer across the entire video.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)):
        print("❌ Model file missing.")
        return

    print("🧠 Loading Model for inspection...")
    
    # 1. Load Vocab size & Model
    with open(VOCAB_PATH, 'r') as f:
        vocab_map = json.load(f)
    
    # Initialize Model
    # Important: map_location ensures we can load GPU weights on CPU if needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNEncoder(num_classes=len(vocab_map), pretrain_mode=True)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- PART A: First Layer Filters (Weights) ---
    print("   • Visualizing First Layer Filters (Conv3d)...")
    filters = model.frontend[0].weight.data.cpu().numpy()
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8)) # Show 32 filters
    fig.suptitle('Learned Filters - Layer 1 (Middle Time Frame)', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(filters):
            # Take the middle frame of the 3D kernel (Depth=1)
            img = filters[i, 0, 1, :, :]
            # Normalize to 0-1 for display
            img = (img - img.min()) / (img.max() - img.min())
            ax.imshow(img, cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # --- PART B: Last Conv Layer Output (Full Video Sequence) ---
    print("   • Visualizing Last Convolutional Layer Activation (Stage 3)...")
    
    # 1. Setup Dataset just to get one valid sample
    # We create a dummy vocab class because the dataset expects it
    class MockVocab:
        def text_to_id(self, x): return 0
    
    try:
        # Load one sample video
        ds = SingleWordDataset(DATA_DIR, LABELS_FILE, vocab=MockVocab(), grayscale=True)
        # We take index 0 (or any other index that works)
        sample_video, _ = ds[0] 
        # sample_video shape: (Channels=1, Time=T, H=88, W=88)
        
        # Add batch dimension -> (1, 1, T, 88, 88)
        input_tensor = sample_video.unsqueeze(0).to(device)
        
    except Exception as e:
        print(f"❌ Could not load sample video: {e}")
        return

    # 2. Register Hook to capture output of stage3
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Hook on the last ResNet block
    hook_handle = model.stage3.register_forward_hook(get_activation('stage3'))

    # 3. Run Forward Pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    hook_handle.remove() # Cleanup hook

    # 4. Process the Output
    # The model processes frames independently in the backbone.
    # Output of stage3 is (Batch * Time, Channels=256, H=6, W=6)
    # We need to reshape it back to (Time, Channels, H, W)
    
    feature_map = activations['stage3'] # (T, 256, 6, 6)
    num_time_steps = feature_map.shape[0]
    
    # Calculate Heatmap: Average across all 256 channels to see "where the model looks"
    # Shape becomes (T, 6, 6)
    heatmaps = feature_map.mean(dim=1).cpu().numpy() 
    
    # Get original frames for comparison (T, 88, 88)
    original_frames = input_tensor[0, 0].cpu().numpy()

    # 5. Plot a Sequence of Frames
    # Select 'num_frames_to_show' indices evenly spaced across the video
    indices = np.linspace(0, num_time_steps - 1, num_frames_to_show, dtype=int)
    
    fig, axes = plt.subplots(2, num_frames_to_show, figsize=(20, 6))
    fig.suptitle(f'Feature Map Activation Sequence (Stage 3 Output)', fontsize=16)

    for i, idx in enumerate(indices):
        # Top Row: Original Frame
        axes[0, i].imshow(original_frames[idx], cmap='gray')
        axes[0, i].set_title(f"Frame {idx}")
        axes[0, i].axis('off')
        
        # Bottom Row: Activation Heatmap
        # We resize the 6x6 heatmap to match the original 88x88 for better visualization
        heatmap_resized = cv2.resize(heatmaps[idx], (88, 88), interpolation=cv2.INTER_CUBIC)
        
        # Overlay: We show the heatmap on top of the original frame (optional, or just side-by-side)
        # Here we show the heatmap using 'inferno' colormap
        im = axes[1, i].imshow(heatmap_resized, cmap='inferno')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"🔎 Analyzing Run: {TARGET_RUN_NAME}")
    
    # 1. Plot Training Graphs
    plot_training_history()
    
    # 2. Plot Confusion Matrix (Top 15 Frequent)
    plot_confusion_matrix(top_k=10)
    
    # 3. Visualize Internals (Filters & Feature Maps)
    visualize_model_internals()
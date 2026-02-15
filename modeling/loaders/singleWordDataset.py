import os
import sys
import cv2
import torch
import json
import numpy as np
from torch.utils.data import Dataset

class SingleWordDataset(Dataset):
    def __init__(self, video_dir, labels_path, vocab, grayscale=True, transform=None):
        """
        Args:
            grayscale (bool): If True, converts to 1 channel. If False, keeps 3 channels (RGB).
        """
        self.video_dir = video_dir
        self.labels_path = labels_path
        self.vocab = vocab
        self.transform = transform
        self.grayscale = grayscale  # Save the preference

        with open(self.labels_path, 'r', encoding='utf-8') as f:
            self.data_map = json.load(f)

        self.file_list = list(self.data_map.keys())

    def __len__(self):
        return len(self.file_list)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- Handling colors ---
            if self.grayscale:
                # If the user wants black and white and the image is in color -> convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Add a channel dimension (H, W, 1)
                frame = np.expand_dims(frame, axis=2) 
            else:
                # If the user wants color
                # OpenCV reads BGR, Python prefers RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(frame)
            
        cap.release()
        return np.array(frames)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        word_text = self.data_map[filename]
        label_id = self.vocab.text_to_id(word_text)
        
        video_path = os.path.join(self.video_dir, filename)
        frames = self.load_video(video_path)

        # Protection: if the video is empty, return one black frame
        if frames is None or len(frames) == 0:
            return None

        # Convert to Tensor: (Time, Height, Width, Channels)
        video_tensor = torch.FloatTensor(frames) / 255.0

        # PyTorch expects: (Channels, Time, Height, Width)
        # Currently we have: (Time, Height, Width, Channels)
        # We need to permute to order the dimensions
        video_tensor = video_tensor.permute(3, 0, 1, 2)

        # Return the video with its original length! The Collate will already arrange the padding
        return video_tensor, torch.tensor(label_id, dtype=torch.long)
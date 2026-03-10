import torch
import json
import os
import decord
from torch.utils.data import Dataset
from decord import VideoReader, cpu

# Set decord to use bridge for faster torch conversion
decord.bridge.set_bridge('torch')

class LipReadingVLMDataset(Dataset):
    def __init__(self, video_dir, json_dir, tokenizer, window_size=24, step_size=4, fps=25, max_seq_len=128):
        self.video_dir = video_dir
        self.json_dir = json_dir
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.step_size = step_size
        self.fps = fps
        self.max_seq_len = max_seq_len
        
        # Professional compromise: A word must have at least 20% of its duration 
        # inside the window to be considered a 'Target'.
        self.overlap_threshold = 0.20 
        
        self.system_prompt = "Analyze the lip movement and transcribe: "
        self.samples = []
        self._build_index()

    def _build_index(self):
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        for j_file in json_files:
            json_path = os.path.join(self.json_dir, j_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            total_frames = data['metadata']['total_frames']
            video_name = j_file.replace('.json', '.mp4')
            video_path = os.path.join(self.video_dir, video_name)

            if os.path.exists(video_path):
                for start_f in range(0, total_frames - self.window_size, self.step_size):
                    self.samples.append({
                        'video_path': video_path,
                        'words_data': data['text']['words'],
                        'start_frame': start_f
                    })
        print(f"[*] Dataset indexed: {len(self.samples)} windows found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start_f = sample['start_frame']
        
        # 1. Load Video Segment using Decord (shape: (1, 24, 88, 88))
        vr = VideoReader(sample['video_path'], ctx=cpu(0))
        indices = list(range(start_f, start_f + self.window_size))
        frames = vr.get_batch(indices) # This returns a tensor (T, H, W, C)

        # Dynamic Channel Check:
        if frames.shape[-1] == 3:
            # If it's RGB (3 channels), take only the first channel to make it grayscale
            # Result: (T, H, W) -> unsqueeze(0) -> (1, T, H, W)
            video_tensor = frames[:, :, :, 0].unsqueeze(0)
        else:
            # If it's already Grayscale (1 channel), just ensure it has the channel dim
            # Some decord versions return (T, H, W). We want (1, T, H, W)
            if frames.ndim == 3:
                video_tensor = frames.unsqueeze(0)
            else:
                # If it's (T, H, W, 1), move channel to front
                video_tensor = frames.permute(3, 0, 1, 2)

        # Final normalization and float conversion
        video_tensor = video_tensor.float() / 255.0

        # 2. Extract Text with Overlap Logic
        start_time = start_f / self.fps
        end_time = (start_f + self.window_size) / self.fps
        words = sample['words_data']
        
        prefix_words = []
        target_words = []

        for w in words:
            w_start, w_end = w['start'], w['end']
            w_duration = w_end - w_start
            
            # Calculate how much of the word is inside the current window
            overlap_start = max(w_start, start_time)
            overlap_end = min(w_end, end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Intersection over Word Duration (IoW)
            overlap_ratio = overlap_duration / w_duration if w_duration > 0 else 0

            if overlap_ratio >= self.overlap_threshold:
                # Word is a Target (significant part is in the video)
                target_words.append(w['word'])
            elif w_end <= start_time:
                # Word finished completely before the window
                prefix_words.append(w['word'])
            # Note: Words that have < 20% overlap and aren't in prefix are effectively ignored
            # to avoid confusing the model with tiny fragments.


        # 3. Final Formatting
        # Define a special string as a placeholder
        video_placeholder = "<video_source>"

        # New Professional Prompt Structure
        prefix_text = f"Instruction: {self.system_prompt}\nInput: {video_placeholder}\nContext: {' '.join(prefix_words[-15:])}\nTranscription: "
        target_text = " ".join(target_words) + self.tokenizer.eos_token

        # Tokenize
        prompt_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        
        input_ids = torch.tensor(prompt_ids + target_ids)
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100 # Mask prompt

        # Padding
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        else:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        return {
            "video": video_tensor,
            "input_ids": input_ids.long(),
            "labels": labels.long(),
            "prompt_len": len(prompt_ids)
        }
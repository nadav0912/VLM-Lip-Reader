import torch
import torch.nn as nn
import torch.nn.functional as F

"""
LIP-READING CNN ENCODER ARCHITECTURE
========================================

Input Video: (Batch, Channels=1, Frames=N, Width=88, Height=88)
--------------------------------------------------------------
               |
       [ 3D FRONT-END ]       <-- Temporal Motion Extraction
       (Conv3d 3x5x5, Stride 1x2x2, BN, ReLU, MaxPool3d)
       - Uses Kernel=3 in Time to capture motion (Pre/Current/Post frames).
       - Stride=1 in Time ensures we don't skip frames (Max Resolution).
               |
               v
       Shape: (Batch, 64, Frames, 22, 22)
               |
       [ FLATTEN TIME ]       <-- Prepare for 2D Spatial Processing
       - Transposes and views the tensor to process each frame independently.
       Shape: (Batch * Frames, 64, 22, 22)
               |
               v
    +-----------------------+
    |   STAGE 1: 64 ch      |
    | [VisualRefiner]       | <-- Identity Residual (x + f(x))
    +-----------------------+     Refines features at 22x22 resolution.
               |
               v
    +-----------------------+
    |   STAGE 2: 128 ch     |
    | [FeatureCompressor]   | <-- Projection Residual (1x1 Conv, Stride 2)
    | [VisualRefiner]       |     Downsamples spatially: 22x22 -> 11x11.
    +-----------------------+
               |
               v
    +-----------------------+
    |   STAGE 3: 256 ch     |
    | [FeatureCompressor]   | <-- Projection Residual (1x1 Conv, Stride 2)
    | [VisualRefiner]       |     Downsamples spatially: 11x11 -> 6x6.
    +-----------------------+
               |
               v
       [ GLOBAL AVG POOL ]    <-- Spatial Reduction (Height/Width -> 1x1)
       - Turns each 6x6 frame into a single 256-dim feature vector.
               |
               v
       [ PRE-TRAIN / VLM ]    <-- Dynamic Output Logic

    1. Pre-training Mode (Word Classification):
       - Temporal Mean Pool: Averages all frame vectors into ONE word vector.
       - Linear Classifier: Predicts the specific word (e.g., "Hello").
    
    2. VLM Mode (LLM Connection):
       - Visual Adapter: Projects each frame vector to LLM space (1536 dim).
       - Output: (Batch, Frames, 1536) - A sequence of visual tokens.

==============================================================================

DESIGN NOTES & ARCHITECTURAL CHOICES:
-------------------------------------
1. HIGH TEMPORAL RESOLUTION:
   Given the dataset median of 3.5 frames/word, we use a Temporal Stride of 1.
   This ensures we produce one visual token per input frame, preventing 
   information loss on extremely short words.

2. HYBRID 3D/2D APPROACH:
   3D Convolution at the front-end captures short-term lip dynamics (motion),
   while the 2D ResNet-style backbone extracts deep spatial features (shape).

3. ADAPTIVE POOLING:
   Using AdaptiveAvgPool2d makes the model resolution-independent. It can 
   handle different input sizes (e.g., 88x88 or 112x112) without 
   breaking the final Linear layers.

4. MEMORY OPTIMIZATION:
   - bias=False: Used before BatchNorm layers as BN cancels out the bias.
   - inplace=True: Used in ReLU to reduce VRAM usage during training.
==============================================================================
"""

# 2D Residual Block for Identity Connection
class IdentityResidualBlock(nn.Module):
    """
    Residual block for 2D convolutional layers when input and output channels are the same.
    Contains 2 convolutional layers with batch normalization and ReLU activation function after each one.
    * The second ReLU is used to add the residual connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return torch.relu(self.conv(x) + x)


# 2D Residual Block for Projection Connection
class ProjectionResidualBlock(nn.Module):
    """
    Residual block for 2D convolutional layers when input and output channels are different.
    Shorcut path is one convolution layer to project the input to the output channels so it can be added to the main path.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The processing path - reduces resolution (stride=2) and increases channels
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # Shortcut path to match dimensions for the residual connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return torch.relu(self.main_path(x) + self.shortcut(x))


class VisualAdapter(nn.Module):
    # Adapter to project the CNN features to the LLM space.
    def __init__(self, cnn_dim=256, llm_dim=1536):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(cnn_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.adapter(x)


# --- Main Model ---
class CNNEncoder(nn.Module):
    def __init__(self, frames=24, llm_embed_dim=1536, num_classes=500, pretrain_mode=True):
        super().__init__()
        self.pretrain_mode = pretrain_mode
        
        # 1. 3D Front-end
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # 2. 2D Backbone
        self.stage1 = IdentityResidualBlock(64)
        self.transition1 = ProjectionResidualBlock(64, 128)
        self.stage2 = IdentityResidualBlock(128)
        self.transition2 = ProjectionResidualBlock(128, 256)
        self.stage3 = IdentityResidualBlock(256)

        # 3. Aggregation & Adapter
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling to reduce spatial dimensions to 1x1
        self.adapter = VisualAdapter(cnn_dim=256, llm_dim=llm_embed_dim)

        # 4. Classifier (Pretrain Mode)
        if self.pretrain_mode:
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(256, num_classes)

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        # Initialize weights with Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 1, T, 88, 88)
        b, c, t, h, w = x.shape
        
        x = self.frontend(x) # (B, 64, T, 22, 22)
        
        # Prepare for 2D: Flatten Time into Batch
        t_new = x.size(2) 
        x = x.transpose(1, 2).contiguous().view(-1, 64, x.size(3), x.size(4))
        
        # Backbone
        x = self.stage1(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        
        # Pooling to Vector
        x = self.gap(x).view(x.size(0), -1) # (B*T, 256)
        
        # Pretrain Mode - Classifier to predict a word.
        if self.pretrain_mode:
            x = x.view(b, t_new, -1).mean(dim=1) # (B, 256)
            x = self.dropout(x)
            x = self.classifier(x)
            return x

        # Fine-tuning Mode - Adapter to project the CNN features to the LLM space.
        else:
            # Projection to LLM Space
            x = self.adapter(x) # (B*T, 1536)
            
            # Reshape back to Sequence
            x = x.view(b, t_new, -1) # (B, T, 1536)
            
            return x


if __name__ == "__main__":
    # Test the model
    model = CNNEncoder()
    dummy_video = torch.randn(2, 1, 20, 88, 88)
    output = model(dummy_video)
    print(f"Final Output Shape for LLM: {output.shape}") 
    # Output: (2, 20, 1536)
import torch
import torch.nn as nn
import numpy as np

class GPSPredictor(nn.Module):
    def __init__(self, feat_model, input_dim=8448, hidden_dim=2048, freeze_backbone=False):
        """
        Args:
            dino_salad_model: The pre-instantiated DINOv2 + SALAD model.
            hidden_dim: Size of the hidden layer in the MLP.
            freeze_backbone: If True, locks weights of DINO model to save VRAM and prevent overfitting.
        """
        super(GPSPredictor, self).__init__()
        self.feat_model = feat_model
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()  # Freeze BatchNorm stats
            print("Backbone frozen: Gradients will not be calculated.")

        # Block 1: Compress high-dim features
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Block 2: Refinement
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Block 3: Output (Lat, Lon)
        # We output 2 values. 
        self.output = nn.Linear(hidden_dim // 2, 2)
        
        # Optional: Initialize weights for better convergence
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        pred = self.output(x)
        
        # Force output to be in range [-1, 1] using Tanh
        # This matches our normalized GPS labels
        return torch.tanh(pred)
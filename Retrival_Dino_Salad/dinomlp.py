import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.amp import autocast
from tqdm import tqdm

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
            for p in self.feat_model.parameters():
                p.requires_grad = False
            self.feat_model.eval()


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
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.feat_model.eval()

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.feat_model(x)
        else:
            x = self.feat_model(x)
            
        x = self.layer1(x)
        x = self.layer2(x)
        pred = self.output(x)
        
        return torch.tanh(pred) / 2 + 0.5 # Scale to [0, 1]
    

    @torch.no_grad()
    def predict_gps(self, images, device="cuda", return_numpy=True):
        """
        images: (B, 3, H, W) or (3, H, W)

        Returns:
          - if input was (3,H,W): shape (2,)  (lat_norm, lon_norm)
          - if input was (B,3,H,W): shape (B,2)
        """
        single = (images.ndim == 3)
        if single:
            images = images.unsqueeze(0)  # [1,3,H,W]

        images = images.to(device)

        # IMPORTANT: put model in eval mode for inference (dropout/bn behavior)
        was_training = self.training
        self.eval()

        preds_norm = self.forward(images)  # [B,2]

        # restore mode (optional but nice)
        if was_training:
            self.train(True)

        if return_numpy:
            out = preds_norm.detach().cpu().numpy()
            return out[0] if single else out
        else:
            return preds_norm[0] if single else preds_norm

def train_dinomlp(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=2e-4,
    weight_decay=1e-2,
    smoothl1_beta=0.01,
    max_grad_norm=1.0,
    use_amp=True,
    device="cuda"
):
    criterion = nn.SmoothL1Loss(beta=smoothl1_beta)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)


    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss, n = 0.0, 0
        desc = "Train" if train else "Val"

        for imgs, gps in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            gps  = gps.to(device, non_blocking=True).float() # [B, 2] in range [0, 1]

            pred = model(imgs) 
            loss = criterion(pred, gps)

            if train:
                opt.zero_grad(set_to_none=True)         
                loss.backward()
                opt.step()

            bs = imgs.size(0)
            total_loss += float(loss.detach()) * bs
            n += bs

        return total_loss / max(n, 1)

    # --- Main Loop ---
    for ep in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        
        with torch.no_grad():
            val_loss = run_epoch(val_loader, train=False)
            
        print(f"Epoch {ep:02d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    return model


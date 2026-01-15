import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
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

    def forward(self, x):
        x = self.feat_model(x)
        x = self.layer1(x)
        x = self.layer2(x)
        pred = self.output(x)
        
        return torch.tanh(pred) / 2 + 0.5 # Scale to [0, 1]

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

    scaler = GradScaler(enabled=(use_amp and device == "cuda"))

    def run_epoch(loader, train: bool):
        model.train(train)
        total, n = 0.0, 0

        for imgs, gps in tqdm(loader, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            gps  = gps.to(device, non_blocking=True).float()  # expected (B,2) in [0,1]

            with autocast(enabled=scaler.is_enabled()):
                pred = model(imgs)
                loss = criterion(pred, gps)

            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()

            bs = imgs.size(0)
            total += float(loss.detach()) * bs
            n += bs

        return total / max(n, 1)

    for ep in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        with torch.no_grad():
            val_loss = run_epoch(val_loader, train=False)
        print(f"Epoch {ep:02d}/{epochs} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

    return model
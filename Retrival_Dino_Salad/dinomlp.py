import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.amp import autocast
from tqdm import tqdm
import os

class GPSPredictor(nn.Module):
    def __init__(self, feat_model, input_dim=8448, hidden_dim=2048, freeze_backbone=True):
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
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Block 2: Refinement
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
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
        return pred

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
        preds_norm = torch.clamp(preds_norm, 0.0, 1.0)

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
    weight_decay=1e-4,
    smoothl1_beta=0.01,
    device="cuda"
):
    criterion = nn.MSELoss()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)


    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss, n = 0.0, 0
        desc = "Train" if train else "Val"



        pbar = tqdm(loader, desc=desc, leave=False)
        for batch_idx, (imgs, gps) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            gps  = gps.to(device, non_blocking=True).float() # [B, 2] in range [0, 1]

            pred = model(imgs) 
            loss = criterion(pred, gps)

            if train:
                opt.zero_grad(set_to_none=True)         
                loss.backward()
                opt.step()

            # --- COLLAPSE CHECK ---
            # Every 10 batches, check if predictions are identical
            # Calculate standard deviation across the batch (dim=0)
            # If std is close to 0, all predictions in the batch are the same.
            lat_std = pred[:, 0].std().item()
            lon_std = pred[:, 1].std().item()

            # Check the GROUND TRUTH spread
            true_lat_std = gps[:, 0].std().item()
            true_lon_std = gps[:, 1].std().item()


            
            # Update the progress bar with the spread
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "pred_std": f"({lat_std:.4f}, {lon_std:.4f})",
                "gt_std": f"({true_lat_std:.4f}, {true_lon_std:.4f})",
            })

            # STOP EARLY if true collapse is detected
            if lat_std < 1e-4 and lon_std < 1e-4 and batch_idx > 5:
                tqdm.write(f"!! WARNING: Mode Collapse Detected at Batch {batch_idx} !!")
                tqdm.write(f"First 3 preds: \n{pred[:3].detach().cpu().numpy()}")
            # ----------------------

            bs = imgs.size(0)
            total_loss += float(loss.detach()) * bs
            n += bs

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


def get_dino_mlp(mlp_head_path, train_loader, val_loader) -> GPSPredictor:
    dino = torch.hub.load("serizba/salad", "dinov2_salad")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dinoMLP_model = GPSPredictor(dino).to(device)
    print("trainable params:", sum(p.numel() for p in dinoMLP_model.parameters() if p.requires_grad))




    if os.path.exists(mlp_head_path):
        print(f"Found MLP head at {mlp_head_path}! Loading...")
        dinoMLP_model.load_state_dict(torch.load(mlp_head_path), strict=False)
    else:
        dinoMLP_model = train_dinomlp(dinoMLP_model, train_loader, val_loader, epochs=20)
        head_weights = {k: v for k, v in dinoMLP_model.state_dict().items() if "feat_model" not in k}
        torch.save(head_weights, mlp_head_path)
        print(f"Model weights saved to {mlp_head_path}")
        
    return dinoMLP_model



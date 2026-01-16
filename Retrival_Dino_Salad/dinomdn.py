import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import GPSRectNorm
from torch.optim import AdamW
from tqdm import tqdm
# -----------------------------
# Building blocks: strong head
# -----------------------------
class SwiGLU(nn.Module):
    """SwiGLU / gated MLP: Linear -> split -> silu gate -> multiply."""
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_hidden * 2)
        self.proj = nn.Linear(dim_hidden, dim_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        x = F.silu(b) * a
        x = self.drop(x)
        return self.proj(x)


class ResGatedBlock(nn.Module):
    """Pre-LN residual block with gated MLP + dropout."""
    def __init__(self, dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim, dim * hidden_mult, dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ff(self.ln(x)))


# -----------------------------
# MDN math (full covariance 2D)
# -----------------------------
def mdn_nll_fullcov_2d(target, pi_logits, mu, L_params,
                       min_diag=1e-4, max_diag=0.2):
    """
    target:   (B,2) in [0,1]
    pi_logits:(B,K)
    mu:       (B,K,2) in [0,1]
    L_params: (B,K,3) representing lower-triangular Cholesky:
              L = [[a, 0],
                   [b, c]]
              with a,c > 0.
    We clamp diag for stability and compute log N using Cholesky.
    """
    B, K, _ = mu.shape

    # mixture weights
    log_pi = F.log_softmax(pi_logits, dim=-1)  # (B,K)

    # unpack Cholesky params
    a_raw = L_params[..., 0]  # (B,K)
    b = L_params[..., 1]      # (B,K)
    c_raw = L_params[..., 2]  # (B,K)

    # enforce positive diagonals (std-like). Use softplus and clamp.
    a = F.softplus(a_raw).clamp(min=min_diag, max=max_diag)
    c = F.softplus(c_raw).clamp(min=min_diag, max=max_diag)

    # build L and compute logdet(Sigma) = 2*sum(log(diag(L)))
    # Sigma = L L^T
    logdet = 2.0 * (torch.log(a) + torch.log(c))  # (B,K)

    # compute y = L^{-1} (x - mu) efficiently for 2D lower-triangular
    x = target.unsqueeze(1)  # (B,1,2)
    dx0 = x[..., 0] - mu[..., 0]  # (B,K)
    dx1 = x[..., 1] - mu[..., 1]  # (B,K)

    # Solve:
    # y0 = dx0 / a
    # y1 = (dx1 - b*y0) / c
    y0 = dx0 / a
    y1 = (dx1 - b * y0) / c

    quad = y0 * y0 + y1 * y1  # Mahalanobis^2 (B,K)

    # log N = -0.5 * (quad + logdet + D*log(2pi))
    logN = -0.5 * (quad + logdet + 2.0 * math.log(2.0 * math.pi))  # (B,K)

    # mixture log-likelihood
    log_prob = torch.logsumexp(log_pi + logN, dim=-1)  # (B,)
    return (-log_prob).mean()


# -----------------------------
# GPSPredictorMDN (strong head)
# -----------------------------
class GPSPredictorMDN(nn.Module):
    """
    Strong MDN head for DINO embeddings -> mixture over (lat_n, lon_n) in [0,1]^2
    Uses:
      - LayerNorm + residual gated blocks (no BatchNorm)
      - Separate towers for pi, mu, covariance
      - Full 2D covariance per component (Cholesky)
    """

    def __init__(
        self,
        feat_model,
        rect_norm: GPSRectNorm,
        input_dim=8448,
        width=1024,
        depth=4,
        K=8,
        dropout=0.15,
        freeze_backbone=True,
    ):
        super().__init__()
        self.feat_model = feat_model
        self.rect_norm = rect_norm
        self.K = int(K)
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for p in self.feat_model.parameters():
                p.requires_grad = False
            self.feat_model.eval()

        # Stem: map DINO embedding -> width
        self.stem = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
        )

        # Shared trunk
        self.trunk = nn.Sequential(*[
            ResGatedBlock(width, hidden_mult=4, dropout=dropout)
            for _ in range(depth)
        ])
        self.trunk_ln = nn.LayerNorm(width)

        # Separate towers (reduce interference)
        self.pi_tower = nn.Sequential(
            ResGatedBlock(width, hidden_mult=2, dropout=dropout),
            nn.LayerNorm(width),
        )
        self.mu_tower = nn.Sequential(
            ResGatedBlock(width, hidden_mult=2, dropout=dropout),
            nn.LayerNorm(width),
        )
        self.cov_tower = nn.Sequential(
            ResGatedBlock(width, hidden_mult=2, dropout=dropout),
            nn.LayerNorm(width),
        )

        # Heads
        self.pi_head = nn.Linear(width, self.K)        # logits
        self.mu_head = nn.Linear(width, self.K * 2)    # means in [0,1]
        self.L_head  = nn.Linear(width, self.K * 3)    # Cholesky params (a,b,c)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.feat_model.eval()

    def _extract_feats(self, images):
        if self.freeze_backbone:
            with torch.no_grad():
                return self.feat_model(images)
        return self.feat_model(images)

    def forward(self, images):
        feats = self._extract_feats(images)      # (B,input_dim)

        x = self.stem(feats)                     # (B,width)
        x = self.trunk(x)
        x = self.trunk_ln(x)

        pi_logits = self.pi_head(self.pi_tower(x))                 # (B,K)

        mu = self.mu_head(self.mu_tower(x)).view(-1, self.K, 2)    # (B,K,2)
        mu = torch.sigmoid(mu)                                     # enforce [0,1]

        L_params = self.L_head(self.cov_tower(x)).view(-1, self.K, 3)  # (B,K,3)

        return pi_logits, mu, L_params

    @torch.no_grad()
    def predict_gps(
        self,
        images,
        device="cuda",
        return_numpy=True,
        strategy="map",     # "map" | "expected" | "topk"
        topk=3,
        return_degrees=True
    ):
        """
        How we choose the final lon/lat:

        - strategy="map" (recommended):
            pick component k* with highest pi_k, return mu_{k*}
        - strategy="expected":
            return sum_k pi_k * mu_k   (can average modes)
        - strategy="topk":
            return top-k candidates (mu_k) + weights (pi_k) so you can choose later
        """
        single = (images.ndim == 3)
        if single:
            images = images.unsqueeze(0)

        images = images.to(device)
        was_training = self.training
        self.eval()

        pi_logits, mu, _ = self.forward(images)
        pi = F.softmax(pi_logits, dim=-1)  # (B,K)

        # helper decode
        def to_out(gps_norm):
            if not return_degrees:
                return gps_norm
            return self.rect_norm.decode_torch(gps_norm[..., 0], gps_norm[..., 1])

        if strategy == "topk":
            k = min(int(topk), self.K)
            w, idx = torch.topk(pi, k=k, dim=-1)  # (B,k)
            idx2 = idx.unsqueeze(-1).expand(-1, -1, 2)
            mu_sel = torch.gather(mu, dim=1, index=idx2)  # (B,k,2)

            out = {
                "gps_norm": mu_sel,
                "weights": w,
                "idx": idx,
            }
            if return_degrees:
                out["gps_deg"] = to_out(mu_sel)

            if was_training:
                self.train(True)

            if return_numpy:
                out = {k: v.detach().cpu().numpy() for k, v in out.items()}
                if single:
                    out = {k: v[0] for k, v in out.items()}
            else:
                if single:
                    out = {k: v[0] for k, v in out.items()}
            return out

        if strategy == "expected":
            gps_norm = (pi.unsqueeze(-1) * mu).sum(dim=1)  # (B,2)
        else:
            # MAP
            idx = torch.argmax(pi, dim=-1)  # (B,)
            gps_norm = mu[torch.arange(mu.size(0), device=mu.device), idx]  # (B,2)

        gps_out = to_out(gps_norm)

        if was_training:
            self.train(True)

        if return_numpy:
            arr = gps_out.detach().cpu().numpy()
            return arr[0] if single else arr
        else:
            return gps_out[0] if single else gps_out

    def mdn_loss(self, gps_norm_target, pi_logits, mu, L_params,
                 min_diag=1e-4, max_diag=0.2):
        return mdn_nll_fullcov_2d(
            target=gps_norm_target,
            pi_logits=pi_logits,
            mu=mu,
            L_params=L_params,
            min_diag=min_diag,
            max_diag=max_diag,
        )

def train_dinomdn(
    model: GPSPredictorMDN,
    train_loader,
    val_loader,
    epochs=10,
    lr=2e-4,
    weight_decay=1e-2,
    max_grad_norm=1.0,
    device="cuda",
):
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=weight_decay)

    def run_epoch(loader, train: bool):
        model.train(train)
        total, n = 0.0, 0

        for imgs, gps in tqdm(loader, desc=("Train" if train else "Val"), leave=False):
            imgs = imgs.to(device, non_blocking=True)
            gps  = gps.to(device, non_blocking=True).float()  # (B,2) in [0,1]

            pi_logits, mu, L_params = model(imgs)
            loss = model.mdn_loss(gps, pi_logits, mu, L_params)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

            bs = imgs.size(0)
            total += float(loss.detach()) * bs
            n += bs

        return total / max(n, 1)

    for ep in range(1, epochs + 1):
        tr = run_epoch(train_loader, True)
        with torch.no_grad():
            va = run_epoch(val_loader, False)
        print(f"Epoch {ep:02d}/{epochs} | Train NLL: {tr:.6f} | Val NLL: {va:.6f}")

    return model

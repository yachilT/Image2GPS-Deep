import os
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
try:
    import faiss  # pip install faiss-cpu
except ImportError as e:
    raise ImportError("FAISS is required. Install with: pip install faiss-cpu") from e


@dataclass
class Match:
    idx: int
    score: float           # cosine similarity (if use_cosine=True) else L2 distance
    gps: Tuple[float, float]


class SaladFaissGPSDB:
    """
    Stores embeddings in a FAISS index and associates each vector with GPS + metadata.

    Cosine mode:
      - normalize embeddings to unit L2
      - use IndexFlatIP
      - similarity = inner product = cosine

    L2 mode:
      - do not normalize (usually)
      - use IndexFlatL2
      - score returned by FAISS is squared L2 distance
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[str] = None,
        normalize: bool = True,
        use_cosine: bool = True,
        embedding_extractor: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        self.model = model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # If using cosine similarity, you almost always want normalization.
        if use_cosine and not normalize:
            raise ValueError("use_cosine=True requires normalize=True (for true cosine similarity).")

        self.normalize = normalize
        self.use_cosine = use_cosine

        # A hook you can override to adapt to SALAD outputs cleanly.
        # It must return a tensor shaped [B, D] (or something reducible to it).
        self.embedding_extractor = embedding_extractor or self._default_embedding_extractor

        self.index: Optional[faiss.Index] = None
        self.gps_list: List[Tuple[float, float]] = []
        self.dim: Optional[int] = None

    # -----------------------------
    # Output handling (FIXED part)
    # -----------------------------
    def _default_embedding_extractor(self, model_out: Any) -> torch.Tensor:
        out = model_out
        return out

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> np.ndarray:
        """
        images: [B,3,H,W]
        returns: np.ndarray [B,D] float32
        """
        if images.ndim != 4:
            raise ValueError(f"images must be [B,3,H,W], got {tuple(images.shape)}")

        images = images.to(self.device, non_blocking=True)
        model_out = self.model(images)

        out = self.embedding_extractor(model_out)


        if out.ndim != 2:
            raise ValueError(f"Expected embeddings reducible to [B,D], got {tuple(out.shape)}")

        out = out.float()
        # spooky
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)

        return out.detach().cpu().numpy().astype(np.float32)

    def _ensure_index(self, dim: int):
        if self.index is not None:
            return

        self.dim = dim
        if self.use_cosine:
            # With L2-normalized vectors, Inner Product == Cosine Similarity
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatL2(dim)

    def add_embeddings(
        self,
        embs: np.ndarray,
        gps: List[Tuple[float, float]],
    ):
        if embs.ndim != 2:
            raise ValueError("embs must be [N,D]")

        n, d = embs.shape
        if len(gps) != n:
            raise ValueError(f"gps length {len(gps)} must match N={n}")

        self._ensure_index(d)
        self.index.add(embs)
        self.gps_list.extend(gps)

    def build_from_loader(self, dataloader):
        """
        Accepts either:
        - DataLoader yielding (images[B,3,H,W], gps[B,2])
        - Dataset yielding   (image[3,H,W], gps[2])
        """
        for batch_i, (images, gps_tensor) in enumerate(tqdm(dataloader)):
            # If dataset gives single sample: image [3,H,W], gps [2]
            if isinstance(images, torch.Tensor) and images.ndim == 3:
                images = images.unsqueeze(0)  # -> [1,3,H,W]

            if isinstance(gps_tensor, torch.Tensor) and gps_tensor.ndim == 1:
                # gps [2] -> [1,2]
                if gps_tensor.numel() != 2:
                    raise ValueError(f"Expected single GPS vector of length 2, got shape {tuple(gps_tensor.shape)}")
                gps_tensor = gps_tensor.unsqueeze(0)

            # Now gps_tensor must be [B,2]
            if gps_tensor.ndim != 2 or gps_tensor.shape[1] != 2:
                raise ValueError(f"Expected gps_tensor shape [B,2], got {tuple(gps_tensor.shape)}")

            gps = [tuple(map(float, row)) for row in gps_tensor.cpu().tolist()]

            embs = self.embed(images)  # returns np.ndarray [B,D]
            self.add_embeddings(embs, gps)
        
    # -----------------------------
    # Query + prediction
    # -----------------------------
    def query_embedding(self, emb: np.ndarray, k: int = 5) -> List[Match]:
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Add data first.")

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(emb, k)
 
        matches: List[Match] = []
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            matches.append(Match(idx=int(idx), score=float(score), gps=self.gps_list[idx]))
        return matches

    def query_image(self, image: torch.Tensor, k: int = 5) -> List[Match]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        emb = self.embed(image)
        return self.query_embedding(emb, k=k)

    def predict_gps(
        self,
        image: torch.Tensor,
        k: int = 5,
        weighted: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[float, float]:
        matches = self.query_image(image, k=k)
        if len(matches) == 0:
            raise RuntimeError("No matches found (index empty?).")

        gps = np.array([m.gps for m in matches], dtype=np.float64)  # [k,2] lat,lon
        if (not weighted) or len(matches) == 1:
            pred = gps.mean(axis=0)
            return float(pred[0]), float(pred[1])

        scores = np.array([m.score for m in matches], dtype=np.float64)

        # ---- 1) Score-based weights (your current logic) ----
        if self.use_cosine:
            w_score = scores - scores.min()
            w_score = w_score + eps
        else:
            w_score = 1.0 / (scores + eps)

        # ---- 2) GPS-consensus weights (leave-one-out + robust scale) ----
        def haversine_m(lat1, lon1, lat2, lon2):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371000.0
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        k_eff = gps.shape[0]
        total = gps.sum(axis=0)

        d = np.empty(k_eff, dtype=np.float64)
        for i in range(k_eff):
            mu_loo = (total - gps[i]) / max(k_eff - 1, 1)
            d[i] = haversine_m(gps[i, 0], gps[i, 1], mu_loo[0], mu_loo[1])

        # robust scale from the distances themselves
        scale = np.median(d) + eps

        # convert distance to weight (closer => larger)
        w_gps = np.exp(- (d / scale) ** 2) + eps

        # ---- 3) Combine and normalize ----
        w = w_score * w_gps
        w = w / (w.sum() + eps)

        pred = (gps * w[:, None]).sum(axis=0)
        return float(pred[0]), float(pred[1])


    # -----------------------------
    # Save / Load
    # -----------------------------
    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Nothing to save (index is None).")

        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        payload = {
            "use_cosine": self.use_cosine,
            "normalize": self.normalize,
            "dim": self.dim,
            "gps": self.gps_list,
        }
        with open(os.path.join(folder, "data.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, folder: str, model: torch.nn.Module, device: Optional[str] = None,
             embedding_extractor: Optional[Callable[[Any], torch.Tensor]] = None) -> "SaladFaissGPSDB":
        with open(os.path.join(folder, "data.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)

        obj = cls(
            model=model,
            device=device,
            normalize=payload["normalize"],
            use_cosine=payload["use_cosine"],
            embedding_extractor=embedding_extractor,
        )
        obj.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        obj.dim = payload["dim"]
        obj.gps_list = [tuple(x) for x in payload["gps"]]
        return obj

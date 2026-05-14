# =============================================================================
# CSOC-SSC v12.4
# Multiscale Criticality-Guided Biomolecular Folding Engine
# =============================================================================
# MIT License — Yoon A Limsuwan 2026
#
# FEATURES
# -----------------------------------------------------------------------------
# • Adaptive Universality Classes
# • Residue-Specific Alpha Fields
# • Contact Diffusion Dynamics
# • SOC / SSC Criticality Engine
# • Dynamic Langevin Thermostat
# • Sparse GPU Physics
# • Multiscale RG Refinement
# • SASA Approximation
# • Distance Matrix Cache
# • T4-Compatible Memory Optimizations
# • Mixed Precision CUDA Support
#
# TARGET
# -----------------------------------------------------------------------------
# Google Colab T4 / A100
# Large-scale de novo folding research
#
# =============================================================================

import os
import math
import time
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# =============================================================================
# METADATA
# =============================================================================

__version__ = "12.4.0"
__author__ = "Yoon A Limsuwan"
__license__ = "MIT"

# =============================================================================
# BIOCHEMISTRY
# =============================================================================

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"

AA_TO_ID = {
    aa: i
    for i, aa in enumerate(AA_VOCAB)
}

HYDROPHOBICITY = {
    'A': 1.8,
    'C': 2.5,
    'D': -3.5,
    'E': -3.5,
    'F': 2.8,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'K': -3.9,
    'L': 3.8,
    'M': 1.9,
    'N': -3.5,
    'P': -1.6,
    'Q': -3.5,
    'R': -4.5,
    'S': -0.8,
    'T': -0.7,
    'V': 4.2,
    'W': -0.9,
    'Y': -1.3
}

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class V124Config:

    device: str = "cuda"

    seed: int = 42

    embedding_dim: int = 128

    hidden_dim: int = 256

    n_layers: int = 4

    n_heads: int = 8

    dropout: float = 0.1

    learning_rate: float = 1e-3

    refinement_steps: int = 600

    gradient_clip: float = 1.0

    use_amp: bool = True

    contact_cutoff: float = 20.0

    sparse_k: int = 32

    weight_bond: float = 30.0

    weight_clash: float = 50.0

    weight_contact: float = 5.0

    weight_sasa: float = 3.0

    weight_criticality: float = 1.0

    use_rg_refinement: bool = True

    rg_levels: int = 3

    rg_factor: int = 4

    base_temperature: float = 300.0

    checkpoint_dir: str = "./v124_checkpoints"

    verbose: int = 1

# =============================================================================
# BACKBONE
# =============================================================================

@dataclass
class Backbone:

    ca: np.ndarray

    seq: str

# =============================================================================
# SEQUENCE EMBEDDING
# =============================================================================

class SequenceEmbedding(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.embedding = nn.Embedding(
            len(AA_VOCAB),
            dim
        )

        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, sequence):

        ids = torch.tensor(
            [
                AA_TO_ID.get(aa, 20)
                for aa in sequence
            ],
            dtype=torch.long,
            device=self.embedding.weight.device
        )

        x = self.embedding(ids)

        x = self.encoder(x)

        return x

# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):

    def __init__(self,
                 dim,
                 max_len=100000):

        super().__init__()

        pe = torch.zeros(max_len, dim)

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim, 2)
            * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):

        n = x.shape[0]

        return x + self.pe[:n]

# =============================================================================
# TRANSFORMER
# =============================================================================

class GeometryTransformer(nn.Module):

    def __init__(self,
                 cfg: V124Config):

        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.n_layers
        )

    def forward(self, x):

        return self.encoder(x)

# =============================================================================
# ADAPTIVE ALPHA FIELD
# =============================================================================

class AdaptiveAlphaPredictor(nn.Module):

    def __init__(self,
                 dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, latent):

        alpha = self.net(latent)

        alpha = torch.sigmoid(alpha)

        alpha = 0.5 + alpha * 2.5

        return alpha.squeeze(-1)

# =============================================================================
# CONTACT DIFFUSION
# =============================================================================

class ContactDiffusion(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self,
                latent,
                coords,
                alpha):

        D = torch.cdist(coords, coords)

        D = D + 1e-6

        ai = alpha.unsqueeze(1)

        aj = alpha.unsqueeze(0)

        a = 0.5 * (ai + aj)

        K = (D ** (-a)) * torch.exp(-D / 12.0)

        K.fill_diagonal_(0)

        K = K / (
            K.sum(dim=-1, keepdim=True)
            + 1e-8
        )

        out = torch.matmul(K, latent)

        return out, K

# =============================================================================
# DISTANCE CACHE
# =============================================================================

class DistanceCache:

    def __init__(self):

        self.cached = None

        self.shape = None

    def compute(self, coords):

        if (
            self.cached is not None
            and
            self.shape == tuple(coords.shape)
        ):

            return self.cached

        D = torch.cdist(coords, coords)

        self.cached = D

        self.shape = tuple(coords.shape)

        return D

# =============================================================================
# CRITICALITY ENGINE
# =============================================================================

class SSCCriticalityEngine:

    def __init__(self):

        self.last_coords = None

    def sigma(self, coords):

        if self.last_coords is None:

            self.last_coords = coords.detach().clone()

            return torch.tensor(
                1.0,
                device=coords.device
            )

        delta = torch.norm(
            coords - self.last_coords,
            dim=-1
        )

        sigma = delta.mean()

        self.last_coords = coords.detach().clone()

        return sigma

    def temperature(self,
                    sigma,
                    base_T=300.0):

        deviation = torch.abs(sigma - 1.0)

        T = base_T * (
            1.0 + 2.0 * deviation
        )

        return torch.clamp(
            T,
            50.0,
            1000.0
        )

# =============================================================================
# SPARSE GRAPH
# =============================================================================

class SparseGraph:

    def __init__(self,
                 coords,
                 cutoff=20.0,
                 k=32):

        self.tree = cKDTree(coords)

        pairs = []

        for i in range(len(coords)):

            idx = self.tree.query_ball_point(
                coords[i],
                cutoff
            )

            idx = [
                j for j in idx
                if j > i
                and abs(i - j) > 3
            ]

            idx = idx[:k]

            for j in idx:

                pairs.append([i, j])

        if len(pairs) == 0:

            pairs = np.zeros((0, 2))

        self.pairs = np.array(
            pairs,
            dtype=np.int64
        )

    def to_torch(self,
                 device):

        return torch.tensor(
            self.pairs,
            dtype=torch.long,
            device=device
        )

# =============================================================================
# PHYSICS
# =============================================================================

def bond_energy(coords,
                weight=30.0):

    dv = coords[1:] - coords[:-1]

    d = torch.norm(dv, dim=-1)

    return weight * torch.mean(
        (d - 3.8) ** 2
    )

def clash_energy(coords,
                 pairs,
                 weight=50.0):

    if len(pairs) == 0:

        return torch.tensor(
            0.0,
            device=coords.device
        )

    dv = (
        coords[pairs[:,0]]
        -
        coords[pairs[:,1]]
    )

    d = torch.norm(dv, dim=-1)

    clash = torch.relu(3.2 - d)

    return weight * torch.mean(
        clash ** 2
    )

def contact_energy(coords,
                   K,
                   weight=5.0):

    D = torch.cdist(coords, coords)

    target = 8.0 * (1.0 - K)

    return weight * torch.mean(
        (D - target) ** 2
    )

def sasa_approximation(coords,
                       seq,
                       D_cache,
                       weight=3.0):

    D = D_cache.compute(coords)

    density = (
        D < 10.0
    ).float().sum(dim=-1)

    burial = 1.0 - torch.exp(
        -density / 20.0
    )

    E = 0.0

    for i, aa in enumerate(seq):

        hydro = HYDROPHOBICITY.get(aa, 0.0)

        if hydro > 0:

            E += hydro * burial[i]

        else:

            E += hydro * (
                1.0 - burial[i]
            )

    return weight * E

# =============================================================================
# RG REFINEMENT
# =============================================================================

class RGRefinement:

    def __init__(self,
                 factor=4):

        self.factor = factor

    def coarse_grain(self,
                     coords):

        n = len(coords)

        nc = (
            n + self.factor - 1
        ) // self.factor

        out = np.zeros(
            (nc, 3),
            dtype=np.float32
        )

        for i in range(nc):

            s = i * self.factor

            e = min(
                (i + 1) * self.factor,
                n
            )

            out[i] = coords[s:e].mean(axis=0)

        return out

    def upsample(self,
                 coarse,
                 n_target):

        x_coarse = np.linspace(
            0,
            n_target - 1,
            len(coarse)
        )

        x_fine = np.arange(n_target)

        out = np.zeros(
            (n_target, 3),
            dtype=np.float32
        )

        for d in range(3):

            cs = CubicSpline(
                x_coarse,
                coarse[:, d]
            )

            out[:, d] = cs(x_fine)

        return out

# =============================================================================
# LANGEVIN OPTIMIZER
# =============================================================================

class SOCLangevinOptimizer(torch.optim.AdamW):

    def __init__(self,
                 params,
                 lr=1e-3):

        super().__init__(params, lr=lr)

        self.dynamic_temperature = 300.0

    @torch.no_grad()
    def step(self, closure=None):

        loss = super().step(closure)

        for group in self.param_groups:

            for p in group["params"]:

                if p.grad is None:
                    continue

                scale = (
                    math.sqrt(
                        self.dynamic_temperature
                        / 300.0
                    )
                    * group["lr"]
                )

                noise = (
                    torch.randn_like(p)
                    * scale
                )

                p.add_(noise)

        return loss

# =============================================================================
# MAIN ENGINE
# =============================================================================

class CSOCSSC_V124(nn.Module):

    def __init__(self,
                 cfg: V124Config):

        super().__init__()

        self.cfg = cfg

        torch.manual_seed(cfg.seed)

        np.random.seed(cfg.seed)

        random.seed(cfg.seed)

        Path(
            cfg.checkpoint_dir
        ).mkdir(exist_ok=True)

        self.device = torch.device(
            cfg.device
            if torch.cuda.is_available()
            else "cpu"
        )

        self.embedding = SequenceEmbedding(
            cfg.embedding_dim
        )

        self.position = PositionalEncoding(
            cfg.embedding_dim
        )

        self.transformer = GeometryTransformer(
            cfg
        )

        self.alpha_predictor = (
            AdaptiveAlphaPredictor(
                cfg.embedding_dim
            )
        )

        self.contact_diffusion = (
            ContactDiffusion()
        )

        self.rg = RGRefinement(
            cfg.rg_factor
        )

        self.to(self.device)

    def log(self, msg):

        if self.cfg.verbose > 0:

            t = time.strftime("%H:%M:%S")

            print(f"[V12.4 {t}] {msg}")

    def encode(self,
               sequence):

        x = self.embedding(sequence)

        x = self.position(x)

        x = x.unsqueeze(0)

        latent = self.transformer(x)

        latent = latent.squeeze(0)

        return latent

    def optimize(self,
                 backbone: Backbone):

        self.log("Encoding sequence")

        latent = self.encode(
            backbone.seq
        )

        alpha = self.alpha_predictor(
            latent
        )

        coords = torch.tensor(
            backbone.ca,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        sparse = SparseGraph(
            backbone.ca,
            cutoff=self.cfg.contact_cutoff,
            k=self.cfg.sparse_k
        )

        sparse_pairs = sparse.to_torch(
            self.device
        )

        D_cache = DistanceCache()

        criticality = SSCCriticalityEngine()

        optimizer = SOCLangevinOptimizer(
            [coords],
            lr=self.cfg.learning_rate
        )

        scaler = GradScaler(
            enabled=self.cfg.use_amp
        )

        self.log("Starting refinement")

        for step in range(
            self.cfg.refinement_steps
        ):

            optimizer.zero_grad()

            with autocast(
                enabled=self.cfg.use_amp
            ):

                latent_diffused, K = (
                    self.contact_diffusion(
                        latent,
                        coords,
                        alpha
                    )
                )

                sigma = criticality.sigma(
                    coords
                )

                T_dynamic = (
                    criticality.temperature(
                        sigma,
                        self.cfg.base_temperature
                    )
                )

                optimizer.dynamic_temperature = (
                    float(T_dynamic)
                )

                E_bond = bond_energy(
                    coords,
                    self.cfg.weight_bond
                )

                E_clash = clash_energy(
                    coords,
                    sparse_pairs,
                    self.cfg.weight_clash
                )

                E_contact = contact_energy(
                    coords,
                    K,
                    self.cfg.weight_contact
                )

                E_sasa = sasa_approximation(
                    coords,
                    backbone.seq,
                    D_cache,
                    self.cfg.weight_sasa
                )

                E_critical = (
                    (sigma - 1.0) ** 2
                    * self.cfg.weight_criticality
                )

                E_latent = (
                    latent_diffused.norm()
                    * 1e-3
                )

                E_total = (
                    E_bond
                    + E_clash
                    + E_contact
                    + E_sasa
                    + E_critical
                    + E_latent
                )

            scaler.scale(E_total).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [coords],
                self.cfg.gradient_clip
            )

            scaler.step(optimizer)

            scaler.update()

            if (
                self.cfg.use_rg_refinement
                and
                step > 0
                and
                step % 200 == 0
            ):

                coarse = self.rg.coarse_grain(
                    coords.detach()
                    .cpu()
                    .numpy()
                )

                refined = self.rg.upsample(
                    coarse,
                    len(coords)
                )

                coords.data = torch.tensor(
                    refined,
                    dtype=torch.float32,
                    device=self.device
                )

            if step % 50 == 0:

                self.log(
                    f"step={step} "
                    f"E={E_total.item():.4f} "
                    f"sigma={sigma.item():.4f} "
                    f"T={T_dynamic.item():.2f}"
                )

        return (
            coords.detach()
            .cpu()
            .numpy()
        )

# =============================================================================
# RMSD
# =============================================================================

def rmsd(a,
         b):

    a = a - a.mean(axis=0)

    b = b - b.mean(axis=0)

    H = a.T @ b

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    ar = a @ R

    return np.sqrt(
        np.mean(
            np.sum(
                (ar - b) ** 2,
                axis=1
            )
        )
    )

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("CSOC-SSC v12.4")
    print("Criticality-Guided Folding Engine")
    print("=" * 80)

    cfg = V124Config(
        refinement_steps=400,
        verbose=1
    )

    model = CSOCSSC_V124(cfg)

    n_res = 300

    coords = (
        np.random.randn(n_res, 3)
        .astype(np.float32)
        * 20.0
    )

    seq = ''.join(
        random.choice(AA_VOCAB[:-1])
        for _ in range(n_res)
    )

    backbone = Backbone(
        ca=coords,
        seq=seq
    )

    start = time.time()

    refined = model.optimize(
        backbone
    )

    elapsed = time.time() - start

    final_rmsd = rmsd(
        coords,
        refined
    )

    print("\nOptimization complete")
    print(f"RMSD: {final_rmsd:.4f} Å")
    print(f"Time: {elapsed:.2f} sec")
    print("=" * 80)

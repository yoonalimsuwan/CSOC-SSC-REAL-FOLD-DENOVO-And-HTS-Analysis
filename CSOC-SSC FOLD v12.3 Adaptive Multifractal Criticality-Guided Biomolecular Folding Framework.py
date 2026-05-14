# =============================================================================
# CSOC-SSC v12.3
# Adaptive Multifractal Criticality-Guided Biomolecular Folding Framework
# =============================================================================
# MIT License — Yoon A Limsuwan 2026
#
# FEATURES
# -----------------------------------------------------------------------------
# [1] Adaptive Alpha Universality Field
# [2] Contact Diffusion Dynamics
# [3] Sparse RG Multiscale Refinement
# [4] Dynamic SOC Langevin Thermostat
# [5] Local + Global Criticality Monitoring
# [6] Hydrophobic Collapse + SASA Approximation
# [7] Sparse GPU Contact Graph
# [8] Differentiable Physics Engine
# [9] Distance Matrix Caching
# [10] Multifractal SOC Dynamics
#
# =============================================================================

import os
import math
import time
import random
import warnings

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np

from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

# =============================================================================
# GLOBALS
# =============================================================================

__version__ = "12.3.0"
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
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
    'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    'X': 0.0
}

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class V123Config:

    device: str = "cuda"

    seed: int = 42

    embedding_dim: int = 192

    hidden_dim: int = 256

    n_layers: int = 4

    n_heads: int = 8

    dropout: float = 0.1

    n_stages: int = 5

    n_iter_per_stage: int = 400

    learning_rate: float = 1e-3

    gradient_clip: float = 1.0

    coarse_factor: int = 4

    contact_cutoff: float = 20.0

    contact_k: int = 64

    sigma_target: float = 1.0

    sigma_tolerance: float = 0.1

    initial_temperature: float = 300.0

    weight_bond: float = 30.0

    weight_angle: float = 8.0

    weight_clash: float = 50.0

    weight_sasa: float = 5.0

    weight_hydrophobic: float = 10.0

    weight_contact: float = 4.0

    weight_criticality: float = 2.0

    weight_rg: float = 1.0

    use_amp: bool = True

    verbose: int = 1

# =============================================================================
# BACKBONE
# =============================================================================

@dataclass
class BackboneFrame:

    ca: np.ndarray

    seq: str

# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

# =============================================================================
# SEQUENCE EMBEDDING
# =============================================================================

class SequenceEncoder(nn.Module):

    def __init__(self,
                 config: V123Config):

        super().__init__()

        self.embedding = nn.Embedding(
            len(AA_VOCAB),
            config.embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_heads,
            batch_first=True,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )

    def tokenize(self,
                 sequence):

        ids = [
            AA_TO_ID.get(aa, 20)
            for aa in sequence
        ]

        return torch.tensor(ids).long()

    def forward(self,
                sequence,
                device):

        tokens = self.tokenize(sequence)

        tokens = tokens.to(device)

        tokens = tokens.unsqueeze(0)

        x = self.embedding(tokens)

        x = self.transformer(x)

        return x.squeeze(0)

# =============================================================================
# ADAPTIVE ALPHA
# =============================================================================

class AdaptiveAlphaField(nn.Module):

    def __init__(self,
                 dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim, dim),

            nn.GELU(),

            nn.Linear(dim, 1)
        )

    def forward(self,
                latent):

        alpha = self.net(latent)

        alpha = torch.sigmoid(alpha)

        alpha = 0.5 + alpha * 2.5

        return alpha.squeeze(-1)

# =============================================================================
# CONTACT DIFFUSION
# =============================================================================

class ContactDiffusion(nn.Module):

    def __init__(self,
                 dim):

        super().__init__()

        self.message = nn.Sequential(

            nn.Linear(dim, dim),

            nn.GELU(),

            nn.Linear(dim, dim)
        )

    def kernel(self,
               D,
               alpha):

        eps = 1e-4

        lam = 12.0

        return (
            (D + eps) ** (-alpha)
        ) * torch.exp(-D / lam)

    def forward(self,
                latent,
                coords,
                alpha):

        D = torch.cdist(coords, coords)

        alpha_ij = (
            alpha[:, None]
            +
            alpha[None, :]
        ) * 0.5

        K = self.kernel(D, alpha_ij)

        K.fill_diagonal_(0.0)

        K = K / (
            K.sum(dim=-1, keepdim=True)
            + 1e-8
        )

        msg = self.message(latent)

        out = torch.matmul(K, msg)

        return latent + out

# =============================================================================
# CRITICALITY ENGINE
# =============================================================================

class CriticalityEngine:

    def __init__(self,
                 target_sigma=1.0):

        self.target_sigma = target_sigma

    def compute_sigma(self,
                      displacement):

        mean_disp = displacement.mean()

        sigma = (
            displacement /
            (mean_disp + 1e-8)
        )

        sigma_global = sigma.mean()

        return sigma_global, sigma

    def temperature(self,
                    sigma_global,
                    base_T):

        delta = abs(
            sigma_global.item()
            -
            self.target_sigma
        )

        return base_T * (1.0 + delta)

# =============================================================================
# DISTANCE CACHE
# =============================================================================

class DistanceCache:

    def __init__(self):

        self.cached = None

        self.last_hash = None

    def compute_hash(self,
                     coords):

        return float(coords.sum().item())

    def get(self,
            coords):

        h = self.compute_hash(coords)

        if (
            self.cached is None
            or
            self.last_hash != h
        ):

            self.cached = torch.cdist(
                coords,
                coords
            )

            self.last_hash = h

        return self.cached

# =============================================================================
# SPARSE GRAPH
# =============================================================================

class SparseGraph:

    def __init__(self,
                 coords,
                 cutoff=20.0,
                 k=64):

        self.coords = coords

        self.cutoff = cutoff

        self.k = k

        self.tree = cKDTree(coords)

    def build(self):

        pairs = []

        for i in range(len(self.coords)):

            neigh = self.tree.query_ball_point(
                self.coords[i],
                self.cutoff
            )

            neigh = [
                j for j in neigh
                if j > i and abs(i - j) > 3
            ]

            neigh = neigh[:self.k]

            for j in neigh:

                pairs.append([i, j])

        if len(pairs) == 0:

            return np.zeros((0, 2), dtype=np.int64)

        return np.array(pairs)

# =============================================================================
# SASA APPROXIMATION
# =============================================================================

class SASAApproximation:

    def __init__(self,
                 cutoff=10.0):

        self.cutoff = cutoff

    def forward(self,
                D):

        density = (
            D < self.cutoff
        ).float().sum(dim=-1)

        burial = 1.0 - torch.exp(
            -density / 15.0
        )

        sasa = 1.0 - burial

        return sasa

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

        out = np.zeros((nc, 3))

        for i in range(nc):

            s = i * self.factor

            e = min(
                (i + 1) * self.factor,
                n
            )

            out[i] = coords[s:e].mean(axis=0)

        return out.astype(np.float32)

    def upsample(self,
                 coarse,
                 n_target):

        x_coarse = np.linspace(
            0,
            n_target - 1,
            len(coarse)
        )

        x_fine = np.arange(n_target)

        out = np.zeros((n_target, 3))

        for d in range(3):

            cs = CubicSpline(
                x_coarse,
                coarse[:, d]
            )

            out[:, d] = cs(x_fine)

        return out.astype(np.float32)

# =============================================================================
# PHYSICS
# =============================================================================

class PhysicsEngine:

    def __init__(self,
                 config: V123Config):

        self.cfg = config

        self.sasa = SASAApproximation()

    def bond_energy(self,
                    coords):

        dv = coords[1:] - coords[:-1]

        d = torch.norm(dv, dim=-1)

        return self.cfg.weight_bond * (
            (d - 3.8) ** 2
        ).mean()

    def angle_energy(self,
                     coords):

        v1 = coords[1:-1] - coords[:-2]

        v2 = coords[2:] - coords[1:-1]

        v1 = F.normalize(v1, dim=-1)

        v2 = F.normalize(v2, dim=-1)

        cosang = (
            v1 * v2
        ).sum(dim=-1)

        target = -0.5

        return self.cfg.weight_angle * (
            (cosang - target) ** 2
        ).mean()

    def clash_energy(self,
                     coords,
                     pairs):

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

        clash = torch.relu(3.0 - d)

        return self.cfg.weight_clash * (
            clash ** 2
        ).mean()

    def hydrophobic_energy(self,
                           coords,
                           sequence,
                           D):

        burial = self.sasa.forward(D)

        E = 0.0

        for i, aa in enumerate(sequence):

            hydro = HYDROPHOBICITY.get(aa, 0.0)

            E += -hydro * burial[i]

        return self.cfg.weight_hydrophobic * E

    def contact_energy(self,
                       D,
                       alpha):

        target = (
            6.0 + 2.0 / alpha
        )

        return self.cfg.weight_contact * (
            (D - target[:, None]) ** 2
        ).mean()

# =============================================================================
# SOC LANGEVIN OPTIMIZER
# =============================================================================

class SOCLangevinOptimizer(torch.optim.AdamW):

    def __init__(self,
                 params,
                 lr=1e-3):

        super().__init__(params, lr=lr)

        self.temperature = 300.0

    def set_temperature(self,
                        T):

        self.temperature = T

    @torch.no_grad()
    def step(self,
             closure=None):

        loss = super().step(closure)

        for group in self.param_groups:

            for p in group["params"]:

                if p.grad is None:
                    continue

                noise_scale = (
                    math.sqrt(
                        self.temperature / 300.0
                    )
                    * group["lr"]
                )

                noise = (
                    torch.randn_like(p)
                    * noise_scale
                )

                p.add_(noise)

        return loss

# =============================================================================
# MAIN ENGINE
# =============================================================================

class CSOCSSC_V123:

    def __init__(self,
                 config: V123Config):

        self.cfg = config

        set_seed(config.seed)

        self.device = torch.device(

            config.device
            if torch.cuda.is_available()
            else "cpu"
        )

        self.encoder = SequenceEncoder(
            config
        ).to(self.device)

        self.alpha_field = AdaptiveAlphaField(
            config.embedding_dim
        ).to(self.device)

        self.contact_diffusion = ContactDiffusion(
            config.embedding_dim
        ).to(self.device)

        self.physics = PhysicsEngine(config)

        self.criticality = CriticalityEngine(
            config.sigma_target
        )

        self.rg = RGRefinement(
            config.coarse_factor
        )

        self.cache = DistanceCache()

    def log(self,
            msg):

        if self.cfg.verbose > 0:

            print(f"[V12.3] {msg}")

    def optimize(self,
                 backbone: BackboneFrame):

        latent = self.encoder(
            backbone.seq,
            self.device
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
            k=self.cfg.contact_k
        )

        pairs = sparse.build()

        pairs = torch.tensor(
            pairs,
            dtype=torch.long,
            device=self.device
        )

        optimizer = SOCLangevinOptimizer(
            [coords],
            lr=self.cfg.learning_rate
        )

        scaler = GradScaler(
            enabled=self.cfg.use_amp
        )

        prev_coords = coords.detach().clone()

        for stage in range(self.cfg.n_stages):

            self.log(f"Stage {stage}")

            for it in range(
                self.cfg.n_iter_per_stage
            ):

                optimizer.zero_grad()

                with autocast(
                    enabled=self.cfg.use_amp
                ):

                    alpha = self.alpha_field(
                        latent
                    )

                    latent = self.contact_diffusion(
                        latent,
                        coords,
                        alpha
                    )

                    D = self.cache.get(coords)

                    E_bond = self.physics.bond_energy(
                        coords
                    )

                    E_angle = self.physics.angle_energy(
                        coords
                    )

                    E_clash = self.physics.clash_energy(
                        coords,
                        pairs
                    )

                    E_hydro = (
                        self.physics
                        .hydrophobic_energy(
                            coords,
                            backbone.seq,
                            D
                        )
                    )

                    E_contact = (
                        self.physics
                        .contact_energy(
                            D,
                            alpha
                        )
                    )

                    displacement = torch.norm(
                        coords - prev_coords,
                        dim=-1
                    )

                    sigma_global, sigma_local = (
                        self.criticality
                        .compute_sigma(
                            displacement
                        )
                    )

                    E_sigma = (
                        self.cfg.weight_criticality
                        *
                        (
                            sigma_global
                            -
                            self.cfg.sigma_target
                        ) ** 2
                    )

                    E_rg = (
                        latent.norm()
                        *
                        self.cfg.weight_rg
                    )

                    E_total = (

                        E_bond +

                        E_angle +

                        E_clash +

                        E_hydro +

                        E_contact +

                        E_sigma +

                        E_rg
                    )

                scaler.scale(E_total).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    [coords],
                    self.cfg.gradient_clip
                )

                T_dynamic = (
                    self.criticality
                    .temperature(
                        sigma_global,
                        self.cfg.initial_temperature
                    )
                )

                optimizer.set_temperature(
                    T_dynamic
                )

                scaler.step(optimizer)

                scaler.update()

                prev_coords = (
                    coords.detach().clone()
                )

                if it % 50 == 0:

                    self.log(
                        f"Iter={it} "
                        f"E={E_total.item():.4f} "
                        f"Sigma={sigma_global.item():.4f} "
                        f"T={T_dynamic:.2f}"
                    )

            if stage < self.cfg.n_stages - 1:

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

        return BackboneFrame(

            ca=coords.detach()
            .cpu()
            .numpy(),

            seq=backbone.seq
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
                axis=-1
            )
        )
    )

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("CSOC-SSC v12.3")
    print("Adaptive Multifractal Criticality Folding Framework")
    print("=" * 80)

    config = V123Config(

        n_stages=4,

        n_iter_per_stage=300,

        verbose=1
    )

    engine = CSOCSSC_V123(config)

    n_res = 512

    coords = (
        np.random.randn(n_res, 3)
        .astype(np.float32)
        * 25.0
    )

    sequence = ''.join(
        random.choice(AA_VOCAB[:-1])
        for _ in range(n_res)
    )

    backbone = BackboneFrame(

        ca=coords,

        seq=sequence
    )

    start = time.time()

    result = engine.optimize(
        backbone
    )

    elapsed = time.time() - start

    final_rmsd = rmsd(
        backbone.ca,
        result.ca
    )

    print("\nOptimization complete")
    print(f"Residues : {n_res}")
    print(f"RMSD     : {final_rmsd:.4f} Å")
    print(f"Time     : {elapsed:.2f} sec")

    print("=" * 80)

# =============================================================================
# CSOC-SSC v15.5
# Multiscale Criticality-Guided Biomolecular Folding Engine
# Full Physics-Enhanced Monolithic Research Edition
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# v15.5 PERFORMANCE UPGRADES:
# -----------------------------------------------------------------------------
# • Fully Vectorized Backbone Reconstruction & Dihedral Math
# • Centralized O(N^2) Distance Matrix Evaluation
# • GPU-Native RG Refinement (via PyTorch F.interpolate)
# • Precomputed Sequence Prior Tensors (Hydrophobicity, Charges, Ramachandran)
# • Removed Differentiable Bottlenecks & CUDA Sync Overheads
# =============================================================================

import os
import math
import time
import random
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# =============================================================================
# METADATA & BIOCHEMISTRY
# =============================================================================

__version__ = "15.5"
__author__ = "Yoon A Limsuwan"
__license__ = "MIT"

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_VOCAB)}

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 
    'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 
    'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3, 'X': 0.0
}

RESIDUE_CHARGE = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.5}

RAMACHANDRAN_PRIORS = {
    'general': {'phi': -60.0, 'psi': -45.0, 'width': 25.0},
    'G': {'phi': -75.0, 'psi': -60.0, 'width': 40.0},
    'P': {'phi': -65.0, 'psi': -30.0, 'width': 20.0},
}

ROTAMER_LIBRARY = {
    'F': [60, 180, -60], 'Y': [60, 180, -60], 'W': [60, 180],
    'L': [60, 180, -60], 'I': [60, -60], 'V': [60, -60], 'M': [60, 180, -60],
}

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class V15Config:
    device: str = "cuda"
    seed: int = 42
    embedding_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    learning_rate: float = 1e-3
    refinement_steps: int = 800
    use_amp: bool = True
    sparse_k: int = 32
    contact_cutoff: float = 20.0
    base_temperature: float = 300.0

    weight_bond: float = 30.0
    weight_clash: float = 60.0
    weight_contact: float = 5.0
    weight_ramachandran: float = 6.0
    weight_torsion: float = 5.0
    weight_hbond: float = 5.0
    weight_rotamer: float = 3.0
    weight_electrostatics: float = 4.0
    weight_solvent: float = 4.0
    weight_criticality: float = 1.0

@dataclass
class Backbone:
    ca: np.ndarray
    seq: str

# =============================================================================
# NEURAL ARCHITECTURE
# =============================================================================

class SequenceEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Embedding(len(AA_VOCAB), dim)
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, seq):
        ids = torch.tensor([AA_TO_ID.get(a, 20) for a in seq], dtype=torch.long, device=self.embedding.weight.device)
        return self.encoder(self.embedding(ids))

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[0]]

class GeometryTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim, dropout=cfg.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def forward(self, x):
        return self.encoder(x)

class AdaptiveAlphaPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, latent):
        alpha = torch.sigmoid(self.net(latent))
        return 0.5 + alpha.squeeze(-1) * 2.5

class ContactDiffusion(nn.Module):
    def forward(self, latent, D, alpha):
        ai = alpha.unsqueeze(1)
        aj = alpha.unsqueeze(0)
        a = 0.5 * (ai + aj)
        K = (D + 1e-6) ** (-a) * torch.exp(-D / 12.0)
        K.fill_diagonal_(0)
        K = K / (K.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(K, latent), K

# =============================================================================
# VECTORIZED GEOMETRY & PHYSICS
# =============================================================================

def compute_dihedral_vectorized(p0, p1, p2, p3):
    """Vectorized calculation of dihedral angles over tensors of shape (N, 3)."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = F.normalize(b1, dim=-1)
    
    v = b0 - (b0 * b1_norm).sum(-1, keepdim=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(-1, keepdim=True) * b1_norm

    x = (v * w).sum(-1)
    y = torch.cross(b1_norm, v, dim=-1)
    y = (y * w).sum(-1)

    return torch.atan2(y, x)

class BackboneReconstruction:
    @staticmethod
    def reconstruct(ca):
        """Vectorized reconstruction to avoid slow python loops."""
        N = torch.zeros_like(ca)
        C = torch.zeros_like(ca)
        
        v = ca[1:] - ca[:-1]
        v = F.normalize(v, dim=-1)
        
        N[1:] = ca[1:] - 1.45 * v
        C[:-1] = ca[:-1] + 1.52 * v
        C[-1] = C[-2]
        
        O = C + torch.tensor([0.0, 1.24, 0.0], device=ca.device)
        return {"N": N, "CA": ca, "C": C, "O": O}

class RamachandranEnergy(nn.Module):
    def forward(self, phi, psi, priors):
        phi_target, psi_target, width = priors
        dphi = (phi - phi_target) / width
        dpsi = (psi - psi_target) / width
        return (dphi**2 + dpsi**2).mean()

class TorsionEnergy(nn.Module):
    def forward(self, angles):
        return (1 + torch.cos(3 * angles)).mean()

class HydrogenBondEnergy(nn.Module):
    def forward(self, atoms):
        D = torch.cdist(atoms["O"], atoms["N"])
        mask = (D > 2.5) & (D < 3.5)
        E = -torch.exp(-((D - 2.95) / 0.3) ** 2)
        return (E * mask.float()).mean()

class DebyeHuckelElectrostatics(nn.Module):
    def __init__(self, dielectric=80.0, kappa=0.1):
        super().__init__()
        self.dielectric = dielectric
        self.kappa = kappa

    def forward(self, D, q):
        qi = q.unsqueeze(1)
        qj = q.unsqueeze(0)
        E = (qi * qj * torch.exp(-self.kappa * D) / (self.dielectric * (D + 1e-6)))
        return E.mean()

class SolventField(nn.Module):
    def forward(self, D, hydro):
        density = (D < 10.0).float().sum(dim=-1)
        burial = 1.0 - torch.exp(-density / 20.0)
        # Vectorized hydrophobicity penalty
        E = torch.where(hydro > 0, hydro * burial, hydro * (1.0 - burial))
        return E.mean()

# =============================================================================
# ENGINES & OPTIMIZATION
# =============================================================================

class SSCCriticalityEngine:
    def __init__(self):
        self.last = None

    def sigma(self, coords):
        if self.last is None:
            self.last = coords.detach().clone()
            return torch.tensor(1.0, device=coords.device)
        delta = torch.norm(coords - self.last, dim=-1)
        sigma = delta.mean()
        self.last = coords.detach().clone()
        return sigma

    def temperature(self, sigma, base=300.0):
        T = base * (1.0 + 2.0 * torch.abs(sigma - 1.0))
        return torch.clamp(T, 50.0, 1000.0)

class GPU_RGRefinement:
    """Replaces SciPy CPU CubicSpline with GPU-native PyTorch interpolation."""
    def __init__(self, factor=4):
        self.factor = factor

    def refine(self, coords):
        n = len(coords)
        nc = n // self.factor
        if nc == 0:
            return coords
            
        # Coarse grain (Mean pool)
        coarse = coords[:nc * self.factor].view(nc, self.factor, 3).mean(dim=1)
        
        # Upsample via GPU Interpolation
        coarse = coarse.permute(1, 0).unsqueeze(0) # Shape: (1, 3, nc)
        refined = F.interpolate(coarse, size=n, mode='linear', align_corners=True)
        return refined.squeeze(0).permute(1, 0) # Shape: (N, 3)

class SOCLangevinOptimizer(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr)
        self.dynamic_temperature = 300.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        scale = math.sqrt(self.dynamic_temperature / 300.0)
        
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    noise = torch.randn_like(p) * (scale * lr)
                    p.add_(noise)
        return loss

# =============================================================================
# MAIN ENGINE
# =============================================================================

class CSOCSSC_V15_5(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.embedding = SequenceEmbedding(cfg.embedding_dim)
        self.position = PositionalEncoding(cfg.embedding_dim)
        self.transformer = GeometryTransformer(cfg)
        self.alpha_predictor = AdaptiveAlphaPredictor(cfg.embedding_dim)
        self.contact_diffusion = ContactDiffusion()
        
        self.rama = RamachandranEnergy()
        self.torsion = TorsionEnergy()
        self.hbond = HydrogenBondEnergy()
        self.electrostatics = DebyeHuckelElectrostatics()
        self.solvent = SolventField()
        self.criticality = SSCCriticalityEngine()
        self.rg = GPU_RGRefinement()
        
        self.to(self.device)
        
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    def precompute_sequence_features(self, seq):
        """Build tensor constants once to avoid rebuilding inside the backward graph."""
        q = torch.tensor([RESIDUE_CHARGE.get(a, 0.0) for a in seq], device=self.device)
        hydro = torch.tensor([HYDROPHOBICITY.get(a, 0.0) for a in seq], device=self.device)
        
        # Pre-compute static rotamer energy
        rot_E = sum(1.0 / len(ROTAMER_LIBRARY[aa]) for aa in seq if aa in ROTAMER_LIBRARY)
        rot_E = torch.tensor(rot_E / len(seq), device=self.device)
        
        # Ramachandran priors
        phi_t, psi_t, widths = [], [], []
        for aa in seq:
            p = RAMACHANDRAN_PRIORS.get(aa, RAMACHANDRAN_PRIORS['general'])
            phi_t.append(p['phi'])
            psi_t.append(p['psi'])
            widths.append(p['width'])
            
        rama_priors = (
            torch.tensor(phi_t, device=self.device),
            torch.tensor(psi_t, device=self.device),
            torch.tensor(widths, device=self.device)
        )
        return q, hydro, rot_E, rama_priors

    def encode(self, sequence):
        x = self.embedding(sequence)
        x = self.position(x).unsqueeze(0)
        return self.transformer(x).squeeze(0)

    def optimize(self, backbone):
        latent = self.encode(backbone.seq)
        alpha = self.alpha_predictor(latent)
        
        # Precompute static tensors
        seq_q, seq_hydro, E_rotamer_static, rama_priors = self.precompute_sequence_features(backbone.seq)

        coords = torch.tensor(backbone.ca, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = SOCLangevinOptimizer([coords], lr=self.cfg.learning_rate)
        scaler = GradScaler(enabled=self.cfg.use_amp)

        for step in range(self.cfg.refinement_steps):
            optimizer.zero_grad()

            with autocast(enabled=self.cfg.use_amp):
                # Centralized distance matrix computed ONCE per step
                D = torch.cdist(coords, coords)
                
                latent_diffused, K = self.contact_diffusion(latent, D, alpha)

                sigma = self.criticality.sigma(coords)
                T = self.criticality.temperature(sigma, self.cfg.base_temperature)
                optimizer.dynamic_temperature = float(T)

                atoms = BackboneReconstruction.reconstruct(coords)

                # Vectorized Dihedrals
                phi = torch.zeros(len(coords), device=self.device)
                psi = torch.zeros(len(coords), device=self.device)
                
                if len(coords) > 2:
                    phi[1:-1] = compute_dihedral_vectorized(atoms["C"][:-2], atoms["N"][1:-1], atoms["CA"][1:-1], atoms["C"][1:-1])
                    psi[1:-1] = compute_dihedral_vectorized(atoms["N"][1:-1], atoms["CA"][1:-1], atoms["C"][1:-1], atoms["N"][2:])
                
                phi = phi * 180.0 / math.pi
                psi = psi * 180.0 / math.pi

                # Energy calculations leveraging D matrix and precomputed tensors
                E_rama = self.rama(phi, psi, rama_priors)
                E_torsion = self.torsion(phi)
                E_hbond = self.hbond(atoms)
                E_electro = self.electrostatics(D, seq_q)
                E_solvent = self.solvent(D, seq_hydro)
                E_rotamer = E_rotamer_static

                # Bond energy
                dv = coords[1:] - coords[:-1]
                d = torch.norm(dv, dim=-1)
                E_bond = ((d - 3.8) ** 2).mean()

                # Clash energy
                clash = torch.relu(3.2 - D)
                E_clash = (clash ** 2).mean()

                # Contact & Criticality
                E_contact = ((D - 8.0 * (1.0 - K)) ** 2).mean()
                E_critical = (sigma - 1.0) ** 2

                E_total = (
                    self.cfg.weight_bond * E_bond +
                    self.cfg.weight_clash * E_clash +
                    self.cfg.weight_contact * E_contact +
                    self.cfg.weight_ramachandran * E_rama +
                    self.cfg.weight_torsion * E_torsion +
                    self.cfg.weight_hbond * E_hbond +
                    self.cfg.weight_rotamer * E_rotamer +
                    self.cfg.weight_electrostatics * E_electro +
                    self.cfg.weight_solvent * E_solvent +
                    self.cfg.weight_criticality * E_critical
                )

            scaler.scale(E_total).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 50 == 0:
                print(f"[v15.5] step={step:03d} E={E_total.item():.4f} rama={E_rama.item():.4f} "
                      f"hbond={E_hbond.item():.4f} electro={E_electro.item():.4f} T={T.item():.2f}")

            # GPU-Native RG Refinement (avoids CPU/NumPy sync bottlenecks)
            if step > 0 and step % 200 == 0:
                with torch.no_grad():
                    coords.data = self.rg.refine(coords)

        return coords.detach().cpu().numpy()

# =============================================================================
# METRICS & RUNNER
# =============================================================================

def rmsd(a, b):
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    return np.sqrt(np.mean(np.sum(((a @ R) - b) ** 2, axis=1)))

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"CSOC-SSC v{__version__} - Performance Optimized")
    print("Full Physics-Enhanced Folding Engine")
    print("=" * 80)

    cfg = V15Config()
    model = CSOCSSC_V15_5(cfg)

    n_res = 300
    coords = (np.random.randn(n_res, 3).astype(np.float32) * 20.0)
    seq = ''.join(random.choice(AA_VOCAB[:-1]) for _ in range(n_res))

    backbone = Backbone(ca=coords, seq=seq)

    start = time.time()
    refined = model.optimize(backbone)
    elapsed = time.time() - start

    final_rmsd = rmsd(coords, refined)

    print("\nOptimization complete")
    print(f"RMSD: {final_rmsd:.4f} Å")
    print(f"Time: {elapsed:.2f} sec")
    print("=" * 80)

#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC v30.6 — Diffusion Module for Protein Backbone Generation
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# This module implements a Denoising Diffusion Probabilistic Model (DDPM)
# for protein backbone coordinates (CA atoms). It can be used as a
# replacement for the EGNN decoder in CSOC‑SSC v30.1/v30.1.1 to generate
# more accurate initial structures.
#
# Features:
#   - SE(3)‑equivariant denoising network (EGNN‑based)
#   - Conditioning on sequence embeddings (or MSA embeddings)
#   - Cosine noise schedule
#   - Training & sampling utilities
#   - Integration hooks for CSOC‑SSC refinement
# =============================================================================

import math, os, random, logging, json, glob
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Re‑use existing CSOC‑SSC modules for encoding & EGNN layers
from csoc_v30_1 import (
    V30_1Config,                          # base config (or V30_1_1Config)
    FlashSequenceEncoder,
    MSAEncoder,                           # if MSA is desired
    AdaptiveAlphaField,                   # not used directly but kept for compatibility
    EGNNLayer,
    SinusoidalPositionalEncoding,         # not strictly needed for diffusion
    # Data loading helpers
    RealProteinDataset,
    MSAProteinDataset,
    MultimerPDBFetcher,
    AA_VOCAB, AA_TO_ID,
)
from torch_cluster import radius_graph

warnings.filterwarnings("ignore")

# ──────────────── Logging ────────────────
def setup_logger(name="CSOC‑SSC_V30.6_Diffusion", local_rank=-1):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('[%(asctime)s] [Rank %(process)d] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(h)
    logger.setLevel(logging.INFO if local_rank in [-1, 0] else logging.WARNING)
    return logger

# ──────────────── Diffusion Configuration ────────────────
@dataclass
class V30_6Config(V30_1Config):
    """
    Extends V30_1Config with diffusion‑specific parameters.
    """
    # Diffusion schedule
    diffusion_steps: int = 1000            # total timesteps
    noise_schedule: str = "cosine"        # "cosine" or "linear"
    min_beta: float = 1e-4
    max_beta: float = 0.02
    cosine_s: float = 0.008               # for cosine schedule

    # Denoising network
    denoiser_hidden: int = 128
    denoiser_layers: int = 6
    denoiser_edge_dim: int = 32
    cutoff: float = 15.0

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 100
    use_amp: bool = True
    checkpoint_dir: str = "./v30_6_diffusion_ckpt"

    # Conditioning
    use_msa_conditioning: bool = False    # if True, will use MSA encoder
    condition_dim: int = 256              # from single‑seq or MSA encoder

    # Sampling
    sampling_steps: int = 200             # number of denoising steps (could be less than diffusion_steps)

    def __post_init__(self):
        super().__post_init__()
        # Compute noise schedule
        if self.noise_schedule == "cosine":
            betas = self._cosine_beta_schedule()
        else:
            betas = torch.linspace(self.min_beta, self.max_beta, self.diffusion_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer = False  # we'll store them later
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.diffusion_steps
        t = torch.linspace(0, steps, steps + 1) / steps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, min=0.001, max=0.999)
        return betas

# ──────────────── Denoising Network (EGNN‑based) ────────────────
class DenoisingNetwork(nn.Module):
    """
    SE(3)‑equivariant network that takes:
      - node features: (L, cond_dim) from encoder
      - coordinates: (L, 3) noisy CA positions
      - timestep: scalar t embedded to a vector
    Returns noise vector (L, 3) that should be subtracted to denoise.
    """
    def __init__(self, cfg: V30_6Config):
        super().__init__()
        self.cfg = cfg
        node_dim = cfg.condition_dim
        hidden_dim = cfg.denoiser_hidden
        edge_dim = cfg.denoiser_edge_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Combine node features with time embedding
        self.node_proj = nn.Linear(node_dim + hidden_dim, hidden_dim)

        # EGNN layers (reuse the same EGNNLayer from v30.1, but we instantiate fresh)
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, edge_dim)
            for _ in range(cfg.denoiser_layers)
        ])
        # Final projection to 3D noise vector
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3, bias=False)
        )

    def forward(self, x, h, t):
        """
        x: noisy coordinates (L, 3)
        h: conditioning node embeddings (L, cond_dim)
        t: timestep (scalar or (1,) tensor), same device
        Returns predicted noise (L, 3)
        """
        L = x.shape[0]
        # Time embedding
        t = torch.tensor([t], device=x.device, dtype=torch.float) if not torch.is_tensor(t) else t.float().view(1)
        t_emb = self.time_embed(t)  # (1, hidden_dim)
        # Concatenate node features with time embedding
        t_emb = t_emb.expand(L, -1)
        node_in = torch.cat([h, t_emb], dim=-1)
        node = self.node_proj(node_in)

        # Build edges dynamically
        edge_index = radius_graph(x, r=self.cfg.cutoff, max_num_neighbors=64, flow='source_to_target')
        edge_dist = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=-1)

        # EGNN passes
        for layer in self.egnn_layers:
            node, x = layer(node, x, edge_index, edge_dist)

        # Predict noise
        noise = self.noise_head(node)
        return noise

# ──────────────── Diffusion Process (DDPM) ────────────────
class DiffusionModel:
    def __init__(self, cfg: V30_6Config):
        self.cfg = cfg
        self.device = cfg.device
        # Register buffers
        self.betas = cfg.betas.to(self.device)
        self.alphas = cfg.alphas.to(self.device)
        self.alphas_cumprod = cfg.alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = cfg.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = cfg.sqrt_one_minus_alphas_cumprod.to(self.device)

        self.denoiser = DenoisingNetwork(cfg).to(self.device)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    def p_sample(self, xt, h, t, noise_scale=1.0):
        """Reverse diffusion step: denoise from t to t-1."""
        with torch.no_grad():
            pred_noise = self.denoiser(xt, h, t)
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
            if t > 0:
                alpha_bar_prev = self.alphas_cumprod[t-1]
                sigma_t = noise_scale * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)
                z = torch.randn_like(xt)
            else:
                sigma_t = 0.0
                z = 0.0
            # Equation to compute x_{t-1}
            x_prev = sqrt_recip_alpha * (xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + sigma_t * z
            return x_prev

    @torch.no_grad()
    def sample(self, h, num_residues=None, init_noise=None, num_steps=None, noise_scale=1.0):
        """
        Generate structure from conditioning h.
        h: (L, cond_dim)
        Returns denoised coordinates (L,3).
        """
        if num_steps is None:
            num_steps = self.cfg.sampling_steps
        if init_noise is None:
            xt = torch.randn(num_residues, 3, device=self.device)
        else:
            xt = init_noise
        L = xt.shape[0]
        timesteps = list(range(num_steps-1, -1, -1))  # full reverse
        for t in timesteps:
            xt = self.p_sample(xt, h, t, noise_scale=noise_scale)
        return xt

    def training_loss(self, x0, h):
        """Compute MSE between predicted noise and true noise."""
        batch = x0.shape[0]  # treat batch as separate chain? Actually we process per sample
        # For simplicity, assume x0 is (L,3) for a single chain
        L = x0.shape[0]
        t = torch.randint(0, self.cfg.diffusion_steps, (1,), device=self.device)
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(xt, h, t)
        loss = F.mse_loss(pred_noise, noise)
        return loss

# ──────────────── High‑level Wrapper (compatible with CSOC‑SSC) ────────────────
class CSOCSSC_Diffusion(nn.Module):
    """
    Wraps the encoder (single‑seq or MSA) and the diffusion model.
    Usage:
        model = CSOCSSC_Diffusion(cfg)
        coords = model.sample(seq_ids, msa=msa)   # generate structure from sequence
    """
    def __init__(self, cfg: V30_6Config):
        super().__init__()
        self.cfg = cfg
        # Choose encoder based on conditioning type
        if cfg.use_msa_conditioning:
            self.encoder = MSAEncoder(cfg.msa_dim, n_layers=4, heads=4, out_dim=cfg.condition_dim)
        else:
            self.encoder = FlashSequenceEncoder(cfg.condition_dim, cfg.depth, cfg.heads, cfg.ff_mult)
        self.diffusion = DiffusionModel(cfg)

    def forward_encoder(self, seq_ids=None, msa=None):
        """
        Get conditioning embeddings.
        seq_ids: (1, L) or (B, L) for batch
        msa: (B, N_seq, L, 22) if using MSA
        """
        if self.cfg.use_msa_conditioning and msa is not None:
            h = self.encoder(msa)  # (B, L, cond_dim)
        else:
            h = self.encoder(seq_ids)  # (B, L, cond_dim)
        return h

    @torch.no_grad()
    def sample(self, seq_ids=None, msa=None, init_noise=None, num_steps=None, noise_scale=1.0):
        """Generate CA coordinates for a single protein."""
        h = self.forward_encoder(seq_ids, msa)
        if h.dim() == 3:
            h = h.squeeze(0)  # (L, cond_dim)
        L = h.shape[0]
        coords = self.diffusion.sample(h, num_residues=L, init_noise=init_noise,
                                       num_steps=num_steps, noise_scale=noise_scale)
        return coords  # (L, 3)

    def training_step(self, seq_ids, target_coords, msa=None):
        """Computes diffusion loss for a batch."""
        h = self.forward_encoder(seq_ids, msa)  # (B, L, cond_dim)
        loss = 0.0
        B = h.shape[0]
        for i in range(B):
            hi = h[i]
            x0 = target_coords[i]
            loss += self.diffusion.training_loss(x0, hi)
        return loss / B

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

# ──────────────── Training Routine (example) ────────────────
def train_diffusion(cfg, dataloader, logger):
    model = CSOCSSC_Diffusion(cfg).to(cfg.device)
    optimizer = model.configure_optimizers()
    scaler = GradScaler(enabled=cfg.use_amp)

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for batch in dataloader:
            if cfg.use_msa_conditioning:
                seq_ids, target_coords, msa = batch
                msa = msa.to(cfg.device, non_blocking=True)
            else:
                seq_ids, target_coords = batch
                msa = None
            seq_ids = seq_ids.to(cfg.device, non_blocking=True)
            target_coords = target_coords.to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type=cfg.device.type, enabled=cfg.use_amp):
                loss = model.training_step(seq_ids, target_coords, msa=msa)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if cfg.local_rank in [-1, 0]:
            logger.info(f"Epoch {epoch+1:03d}/{cfg.epochs}  Loss={total_loss/len(dataloader):.4f}")

    if cfg.local_rank in [-1, 0]:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(cfg.checkpoint_dir, "v30_6_diffusion.pt")
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")

# ──────────────── Example usage & CLI (standalone) ────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CSOC‑SSC v30.6 – Diffusion Protein Folding")
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train')
    train_parser.add_argument('--pdb_dir', type=str, required=True)
    train_parser.add_argument('--msa_dir', type=str, default=None)
    train_parser.add_argument('--epochs', type=int, default=100)

    sample_parser = sub.add_parser('sample')
    sample_parser.add_argument('--seq', type=str, required=True)
    sample_parser.add_argument('--checkpoint', type=str, required=True)
    sample_parser.add_argument('--out', type=str, default='diffusion_sample.pdb')
    sample_parser.add_argument('--steps', type=int, default=200)
    sample_parser.add_argument('--msa', type=str, default=None)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = V30_6Config(device=device)
    if args.command == 'train':
        if args.msa_dir:
            cfg.use_msa_conditioning = True
            dataset = MSAProteinDataset(args.pdb_dir, args.msa_dir)
        else:
            dataset = RealProteinDataset(args.pdb_dir)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                                                 pin_memory=True)
        logger = setup_logger()
        train_diffusion(cfg, dataloader, logger)

    elif args.command == 'sample':
        logger = setup_logger()
        # Load model
        model = CSOCSSC_Diffusion(cfg).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        seq = args.seq
        seq_ids = torch.tensor([AA_TO_ID.get(a,20) for a in seq], dtype=torch.long, device=device).unsqueeze(0)
        msa = None
        if args.msa:
            msa_np = np.load(args.msa)
            msa = torch.tensor(msa_np, dtype=torch.float32).unsqueeze(0).to(device)
        coords = model.sample(seq_ids=seq_ids, msa=msa, num_steps=args.steps)
        # Write simple PDB
        with open(args.out, 'w') as f:
            for i in range(coords.shape[0]):
                f.write(f"ATOM  {i+1:5d}  CA  {seq[i] if i<len(seq) else 'X'} A{i+1:4d}    {coords[i,0]:8.3f}{coords[i,1]:8.3f}{coords[i,2]:8.3f}  1.00  0.00           C\n")
            f.write("END\n")
        logger.info(f"Sample saved to {args.out}")

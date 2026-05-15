#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC v27 — Full‑Atom GPU‑Native SOC Folding & Refinement Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# V27 adds full‑atom side‑chain reconstruction and a physics‑based energy
# function that refines both backbone (CA‑trace) and side‑chain torsion angles
# simultaneously on GPU.  All V26 features are preserved.
# =============================================================================

import os, math, time, random, argparse, logging, urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def setup_logger(name="CSOC‑SSC_V27", local_rank=-1):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            '[%(asctime)s] [Rank %(process)d] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'))
        logger.addHandler(h)
    logger.setLevel(logging.INFO if local_rank in [-1, 0] else logging.WARNING)
    return logger

# ──────────────────────────────────────────────────────────────────────────────
# Biochemical constants
# ──────────────────────────────────────────────────────────────────────────────
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_VOCAB)}

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    'X': 0.0
}

RESIDUE_CHARGE = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.5}

RAMACHANDRAN_PRIORS = {
    'general': {'phi': -60.0, 'psi': -45.0, 'width': 25.0},
    'G':       {'phi': -75.0, 'psi': -60.0, 'width': 40.0},
    'P':       {'phi': -65.0, 'psi': -30.0, 'width': 20.0},
}

AA_3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
}

# ──────────────────────────────────────────────────────────────────────────────
# Full‑Atom Topology (internal coordinates for side‑chain heavy atoms)
# ──────────────────────────────────────────────────────────────────────────────
# Each residue template: list of (atom_name, atom_type, parent_idx, bond_len,
# bond_ang, dihedral_ref_idx, dihedral_ang0)
#  * parent_idx = index in the atom list (0 = CB, -1 means unknown – not used)
#  * dihedral_ref_idx: tuple (idx_a, idx_b, idx_c) of atoms that define the
#    dihedral zero when chi=0
#  * dihedral_ang0: the torsion angle (in radians) when chi=0 for that atom.
#    This is the offset so that the chi angle represents rotation from that.
#  * atom_type is a string for LJ parameters.
# We define a default set of LJ parameters (sigma, epsilon) per atom type.
# CB and other atoms will be built using a standard builder.

# Atom types and LJ parameters (united‑atom style, heavy atoms only, Å, kcal/mol)
LJ_PARAMS = {
    'C':   (1.9080, 0.0860),   # carbonyl C
    'CA':  (1.9080, 0.0860),   # alpha C (we don't use it for LJ, but just in case)
    'CB':  (1.9080, 0.0860),   # aliphatic C
    'CG':  (1.9080, 0.0860),
    'CD':  (1.9080, 0.0860),
    'CE':  (1.9080, 0.0860),
    'CZ':  (1.9080, 0.0860),
    'CH2': (1.9080, 0.0860),
    'N':   (1.8240, 0.1700),   # amide N
    'ND':  (1.8240, 0.1700),
    'NE':  (1.8240, 0.1700),
    'NH1': (1.8240, 0.1700),
    'NH2': (1.8240, 0.1700),
    'O':   (1.6612, 0.2100),   # carbonyl O
    'OD':  (1.6612, 0.2100),
    'OE':  (1.6612, 0.2100),
    'OH':  (1.6612, 0.2100),
    'S':   (2.0000, 0.2500),   # sulphur
    'SG':  (2.0000, 0.2500),
}

# Internal coordinates for each residue (based on CHARMM/OPLS, slightly simplified)
# CB is built from N, CA, C. So index0 will be CB.
# For GLY no sidechain, for ALA only CB, for others longer.
# We define template per one-letter code.
# dihedral_ref_idx uses negative indices: -1 = N, -2 = CA, -3 = C, -4 = CB,
# -5... further atoms. We'll map at build time.
RESIDUE_TOPOLOGY = {
    'G': [],   # Gly has no heavy sidechain (only H, which we omit)
    'A': [ # Ala: only CB
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),   # but CB is index 0, we'll handle specially
    ],
    'S': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('OG', 'OH', 1, 1.43, 109.5, (-2,-1,0), 0.0),  # chi1 around CA-CB
    ],
    'C': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('SG', 'SG', 1, 1.81, 109.5, (-2,-1,0), 0.0),  # chi1
    ],
    'V': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG1','CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
        ('CG2','CB', 1, 1.53, 109.5, (-2,-1,0), 2.0),  # offset 120 deg
    ],
    'T': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('OG1','OH', 1, 1.43, 109.5, (-2,-1,0), 0.0),
        ('CG2','CB', 1, 1.53, 109.5, (-2,-1,0), 2.0),
    ],
    'L': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
        ('CD1','CB', 2, 1.53, 109.5, (-1,0,1), 0.0),
        ('CD2','CB', 2, 1.53, 109.5, (-1,0,1), 2.0),
    ],
    'I': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG1','CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
        ('CG2','CB', 1, 1.53, 109.5, (-2,-1,0), 2.0),
        ('CD1','CB', 2, 1.53, 109.5, (-1,0,1), 0.0),
    ],
    'M': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
        ('SD', 'S',  2, 1.81, 109.5, (-1,0,1), 0.0),
        ('CE', 'CB', 3, 1.81, 109.5, (-2,-1,0), 0.0),
    ],
    'F': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
        ('CD1','CB', 2, 1.40, 120.0, (-1,0,1), 0.0),
        ('CD2','CB', 2, 1.40, 120.0, (-1,0,1), 2.0),
        ('CE1','CB', 3, 1.40, 120.0, (2,1,0), 0.0),
        ('CE2','CB', 4, 1.40, 120.0, (2,1,0), 2.0),
        ('CZ', 'CB', 5, 1.40, 120.0, (3,2,1), 0.0),
    ],
    'Y': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
        ('CD1','CB', 2, 1.40, 120.0, (-1,0,1), 0.0),
        ('CD2','CB', 2, 1.40, 120.0, (-1,0,1), 2.0),
        ('CE1','CB', 3, 1.40, 120.0, (2,1,0), 0.0),
        ('CE2','CB', 4, 1.40, 120.0, (2,1,0), 2.0),
        ('CZ', 'CB', 5, 1.40, 120.0, (3,2,1), 0.0),
        ('OH', 'OH', 6, 1.36, 120.0, (4,3,2), 0.0),
    ],
    'W': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.53, 109.5, (-2,-1,0), 0.0),
        ('CD1','CB', 2, 1.40, 120.0, (-1,0,1), 0.0),
        ('CD2','CB', 2, 1.40, 120.0, (-1,0,1), 2.0),
        ('NE1','N',  3, 1.38, 120.0, (2,1,0), 0.0),
        ('CE2','CB', 4, 1.40, 120.0, (2,1,0), 2.0),
        ('CE3','CB', 5, 1.40, 120.0, (2,1,0), 2.0),
        ('CZ2','CB', 6, 1.40, 120.0, (3,2,1), 0.0),
        ('CZ3','CB', 7, 1.40, 120.0, (5,4,2), 0.0),
        ('CH2','CB', 8, 1.40, 120.0, (6,3,2), 0.0),
    ],
    'D': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'C',  1, 1.52, 109.5, (-2,-1,0), 0.0),
        ('OD1','O',  2, 1.25, 120.0, (-1,0,1), 0.0),
        ('OD2','O',  2, 1.25, 120.0, (-1,0,1), 2.0),
    ],
    'E': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.52, 109.5, (-2,-1,0), 0.0),
        ('CD', 'C',  2, 1.52, 109.5, (-1,0,1), 0.0),
        ('OE1','O',  3, 1.25, 120.0, (2,1,0), 0.0),
        ('OE2','O',  3, 1.25, 120.0, (2,1,0), 2.0),
    ],
    'N': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'C',  1, 1.52, 109.5, (-2,-1,0), 0.0),
        ('OD1','O',  2, 1.25, 120.0, (-1,0,1), 0.0),
        ('ND2','N',  2, 1.33, 120.0, (-1,0,1), 2.0),
    ],
    'Q': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.52, 109.5, (-2,-1,0), 0.0),
        ('CD', 'C',  2, 1.52, 109.5, (-1,0,1), 0.0),
        ('OE1','O',  3, 1.25, 120.0, (2,1,0), 0.0),
        ('NE2','N',  3, 1.33, 120.0, (2,1,0), 2.0),
    ],
    'K': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.52, 109.5, (-2,-1,0), 0.0),
        ('CD', 'CB', 2, 1.52, 109.5, (-1,0,1), 0.0),
        ('CE', 'CB', 3, 1.52, 109.5, (-2,-1,0), 0.0),
        ('NZ', 'N',  4, 1.47, 109.5, (-1,0,1), 0.0),
    ],
    'R': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.52, 109.5, (-2,-1,0), 0.0),
        ('CD', 'CB', 2, 1.52, 109.5, (-1,0,1), 0.0),
        ('NE', 'N',  3, 1.46, 109.5, (-2,-1,0), 0.0),
        ('CZ', 'C',  4, 1.33, 125.0, (-1,0,1), 0.0),
        ('NH1','N',  5, 1.33, 120.0, (4,3,2), 0.0),
        ('NH2','N',  5, 1.33, 120.0, (4,3,2), 2.0),
    ],
    'H': [
        ('CB', 'CB', 0, 1.53, 109.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.50, 109.5, (-2,-1,0), 0.0),
        ('ND1','N',  2, 1.38, 120.0, (-1,0,1), 0.0),
        ('CD2','CB', 2, 1.40, 120.0, (-1,0,1), 2.0),
        ('CE1','CB', 3, 1.40, 120.0, (2,1,0), 0.0),
        ('NE2','N',  4, 1.38, 120.0, (2,1,0), 2.0),
    ],
    'P': [
        ('CB', 'CB', 0, 1.53, 104.5, (-1,-2,-3), 0.0),
        ('CG', 'CB', 1, 1.50, 104.5, (-2,-1,0), 0.0),
        ('CD', 'CB', 2, 1.50, 104.5, (-1,0,1), 0.0),
    ],
}

# Maximum number of chi angles per residue (excluding terminal chi for some)
MAX_CHI = 4
# Map residue to number of chi angles we will optimise
RESIDUE_NCHI = {
    'A': 0, 'G': 0, 'S': 1, 'C': 1, 'V': 1, 'T': 1, 'L': 2, 'I': 2,
    'M': 3, 'F': 2, 'Y': 2, 'W': 2, 'D': 2, 'E': 3, 'N': 2, 'Q': 3,
    'K': 4, 'R': 4, 'H': 2, 'P': 2,
}

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class V27Config:
    local_rank: int = int(os.environ.get("LOCAL_RANK", -1))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    dim: int = 256
    depth: int = 6
    heads: int = 8
    ff_mult: int = 4

    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 80
    use_amp: bool = True
    gradient_accumulation_steps: int = 1

    refine_steps: int = 600
    temp_base: float = 300.0
    friction: float = 0.02
    sigma_target: float = 1.0
    avalanche_threshold: float = 0.5
    avalanche_topk: int = 3
    w_avalanche: float = 0.2

    ca_ca_dist: float = 3.8
    clash_radius: float = 3.5
    angle_target_rad: float = 110.0 * math.pi / 180.0

    alpha_mod_bond: float = 0.1
    alpha_mod_angle: float = 0.05
    alpha_mod_rama: float = 0.2
    alpha_mod_clash: float = 0.1
    alpha_mod_hbond: float = 0.1

    # Weights (original)
    w_bond: float = 30.0
    w_angle: float = 15.0
    w_rama: float = 8.0
    w_clash: float = 80.0
    w_hbond: float = 6.0
    w_electro: float = 4.0
    w_solvent: float = 5.0
    w_rotamer: float = 3.0
    w_alpha_entropy: float = 0.5
    w_alpha_smooth: float = 0.1
    w_soc_contact: float = 0.3

    # New full‑atom weights
    w_lj: float = 50.0             # Lennard‑Jones
    w_coulomb: float = 5.0         # Coulomb
    w_torsion: float = 10.0        # chi torsion

    sparse_cutoff: float = 12.0
    kernel_lambda: float = 12.0
    rebuild_interval: int = 100

    use_rg: bool = True
    rg_factor: int = 4
    rg_interval: int = 200

    checkpoint_dir: str = "./v27_ckpt"
    out_pdb: str = "refined_v27.pdb"

# ──────────────────────────────────────────────────────────────────────────────
# PDB Fetcher (same as v26 but now we also need native all‑atom for RMSD)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Backbone:
    ca: np.ndarray
    seq: str
    chain_ids: Optional[np.ndarray] = None
    native_coords: Optional[np.ndarray] = None   # CA only for RMSD

class PDBFetcher:
    @staticmethod
    def fetch_and_parse(pdb_id: str) -> Backbone:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        req = urllib.request.Request(url, headers={'User-Agent': 'CSOC-SSC_V27'})
        ca_coords, seq_list, chain_ids = [], [], []
        chain_map = {}
        try:
            with urllib.request.urlopen(req) as response:
                lines = response.read().decode('utf-8').split('\n')
            for line in lines:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if chain_id not in chain_map:
                        chain_map[chain_id] = len(chain_map)
                    seq_list.append(AA_3_TO_1.get(res_name, 'X'))
                    ca_coords.append([x, y, z])
                    chain_ids.append(chain_map[chain_id])
            coords_arr = np.array(ca_coords, dtype=np.float32)
            seq_str = "".join(seq_list)
            chain_arr = np.array(chain_ids, dtype=np.int32)
            randomized_coords = coords_arr + (np.random.randn(*coords_arr.shape) * 10.0).astype(np.float32)
            return Backbone(ca=randomized_coords, seq=seq_str, chain_ids=chain_arr, native_coords=coords_arr)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch PDB {pdb_id}: {str(e)}")

# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding & Transformer (unchanged from v26)
# ──────────────────────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class FlashGeometryBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Linear(dim * ff_mult, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                      dropout_p=0.1 if self.training else 0.0)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        h = self.proj(attn_out)
        x = self.norm1(x + self.dropout(h))
        y = self.ffn(x)
        x = self.norm2(x + self.dropout(y))
        return x

class FlashSequenceEncoder(nn.Module):
    def __init__(self, dim, depth, heads, ff_mult):
        super().__init__()
        self.embed = nn.Embedding(len(AA_VOCAB), dim)
        self.pos_enc = SinusoidalPositionalEncoding(dim)
        self.layers = nn.ModuleList([FlashGeometryBlock(dim, heads, ff_mult) for _ in range(depth)])

    def forward(self, seq_ids):
        x = self.embed(seq_ids)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return x

class GeometryDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 3))

    def forward(self, latent):
        coords = self.net(latent)
        return coords - coords.mean(dim=1, keepdim=True)

class AdaptiveAlphaField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, latent):
        a = torch.sigmoid(self.net(latent))
        a = 0.5 + 2.5 * a.squeeze(-1)
        return torch.clamp(a, 0.5, 3.0)

# ──────────────────────────────────────────────────────────────────────────────
# CSOC Controller (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class CSOCController:
    def __init__(self):
        self.prev_coords = None

    def sigma(self, coords):
        if self.prev_coords is None:
            self.prev_coords = coords.detach().clone()
            return torch.tensor(1.0, device=coords.device)
        delta = torch.norm(coords - self.prev_coords, dim=-1).mean()
        self.prev_coords = coords.detach().clone()
        return delta

    def temperature(self, sigma, base_T, target):
        dev = (sigma - target) / 0.5
        T = base_T + 2000.0 * torch.sigmoid(dev)
        return torch.clamp(T, base_T * 0.5, 3000.0)

# ──────────────────────────────────────────────────────────────────────────────
# Differentiable RG Refinement (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class DiffRGRefiner:
    def __init__(self, factor=4):
        self.factor = factor

    def forward(self, coords):
        L = coords.shape[0]
        f = self.factor
        m = L // f * f
        if m == 0:
            return coords
        x = coords[:m].permute(1, 0).unsqueeze(0)
        pooled = F.avg_pool1d(x, kernel_size=f, stride=f)
        up = F.interpolate(pooled, size=L, mode='linear', align_corners=True)
        return up.squeeze(0).permute(1, 0)

# ──────────────────────────────────────────────────────────────────────────────
# Sparse Graph Builder (GPU)
# ──────────────────────────────────────────────────────────────────────────────
def build_sparse_graph(coords, cutoff):
    D = torch.cdist(coords, coords)
    triu = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
    mask = (D < cutoff) & triu
    src, dst = torch.where(mask)
    edge_dist = D[src, dst]
    src = torch.cat([src, dst])
    dst = torch.cat([dst, src])
    edge_dist = torch.cat([edge_dist, edge_dist])
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, edge_dist

# ──────────────────────────────────────────────────────────────────────────────
# Full‑Atom Builder (Differentiable)
# ──────────────────────────────────────────────────────────────────────────────
def build_sidechain_atoms(ca, seq, chi_angles):
    """
    ca: (L, 3) tensor
    seq: string of length L
    chi_angles: tensor of shape (L, max_chi) where padding is 0
    Returns:
        all_atom_coords: list of tensors, one per residue, each (N_i, 3)
        all_atom_types: list of lists of atom type strings
    """
    device = ca.device
    L = ca.shape[0]
    # Reconstruct backbone
    v = ca[1:] - ca[:-1]
    v_norm = F.normalize(v, dim=-1, eps=1e-8)
    N = torch.zeros_like(ca)
    C = torch.zeros_like(ca)
    N[1:] = ca[1:] - 1.45 * v_norm
    N[0] = ca[0] - 1.45 * v_norm[0]
    C[:-1] = ca[:-1] + 1.52 * v_norm
    C[-1] = ca[-1] + 1.52 * v_norm[-1]

    all_coords = []
    all_types = []
    for i, aa in enumerate(seq):
        if aa == 'G':
            # only backbone atoms: N, CA, C (we don't include O for energy, but it's built later)
            # We'll store N, CA, C as a minimal set
            res_atoms = torch.stack([N[i], ca[i], C[i]], dim=0)   # (3,3)
            types = ['N', 'CA', 'C']
            all_coords.append(res_atoms)
            all_types.append(types)
            continue

        # Build CB using standard geometry (as in energy_rotamer)
        n_i = N[i]
        ca_i = ca[i]
        c_i = C[i]
        v1 = n_i - ca_i
        v2 = c_i - ca_i
        cb_dir = -(v1 + v2)
        cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
        cb_pos = ca_i + 1.53 * cb_dir

        # Collect local atoms in order: N, CA, C, CB, then sidechain
        local_atoms = [n_i, ca_i, c_i, cb_pos]
        # mapping from negative indices to local positions: -1->N, -2->CA, -3->C, -4->CB
        # further atoms will have indices 4,5,...
        local_types = ['N', 'CA', 'C', 'CB']

        # Get topology for this residue
        topo = RESIDUE_TOPOLOGY.get(aa, [])
        chi_idx = 0
        for (atom_name, atom_type, parent_idx, bond_len, bond_ang_deg, ref_tuple, dihedral_ang0) in topo:
            # parent_idx is index in local list (0 = CB, but we have offset: since local_atoms[3] is CB, parent_idx 0 means local_atoms[3]? Careful.
            # In our topology, we defined parent_idx starting from 0 for CB. So parent_idx = 0 means local_atoms[3].
            # We'll map: parent_local = 3 if parent_idx == 0, else parent_idx+3? Actually, we'll define:
            # The template assumes indices: 0: CB, 1: next atom, etc.
            # So first atom (CB) is index 0, second (if any) index 1, etc.
            # Thus, the absolute index in local list is 3 + parent_idx.
            parent_abs = 3 + parent_idx

            # Determine reference atoms for dihedral
            # ref_tuple contains three indices: a,b,c (using same numbering: 0=CB, 1=next, etc. or negative for backbone)
            # We need to map to local indices.
            def map_ref(idx):
                if idx == -1: return 0  # N
                if idx == -2: return 1  # CA
                if idx == -3: return 2  # C
                if idx == -4: return 3  # CB
                # positive indices: 0=CB, 1=... -> offset 3
                return 3 + idx

            a_idx = map_ref(ref_tuple[0])
            b_idx = map_ref(ref_tuple[1])
            c_idx = map_ref(ref_tuple[2])

            # Get positions
            p_b = local_atoms[b_idx]
            p_c = local_atoms[c_idx]
            # parent position
            p_parent = local_atoms[parent_abs]

            # bond angle in radians
            ang = torch.tensor(bond_ang_deg * math.pi / 180.0, device=device)

            # The dihedral zero angle (ref dihedral) is the dihedral between a,b,c and b,c,parent?
            # Standard: dihedral(a,b,c,d) where d is the new atom placed with bond length, angle, and dihedral offset.
            # We have atoms a,b,c and the parent is the atom we are building? Actually, the parent is the atom to which we attach.
            # In standard IC, atom d is built from a,b,c such that bond distance = |d-c|, angle between b-c and c-d, and dihedral a-b-c-d.
            # Here parent_abs is the atom to attach (c). Wait, we have bond_len from parent? Usually the new atom is attached to parent.
            # Let's redefine: We want to build the new atom attached to parent. So parent is the atom we attach to.
            # We'll set a,b,c as reference atoms where c is the parent. So we need a,b,c such that a-b-c gives a dihedral reference.
            # In the template, parent_idx indicates the atom we attach to (the "c" in a-b-c-d). The ref_tuple should give a,b,c where c == parent_abs.
            # We'll assign: c = parent_abs, b = a neighbour of c, a = a neighbour of b.
            # But we already have a_idx,b_idx,c_idx. Let's assume ref_tuple[2] is the parent (c). Then we'll set:
            # c = local_atoms[map_ref(ref_tuple[2])]   (should equal parent)
            # b = local_atoms[map_ref(ref_tuple[1])]
            # a = local_atoms[map_ref(ref_tuple[0])]
            p_a = local_atoms[a_idx]
            p_b = local_atoms[b_idx]
            # parent is c (the atom we attach to)
            p_c_attach = local_atoms[c_idx]   # should be p_parent
            # if not, we'll trust the mapping

            # Compute bond direction from c to new atom using bond length, angle and dihedral
            # First, get unit vectors
            bc = p_c_attach - p_b
            ab = p_b - p_a
            bc_norm = F.normalize(bc, dim=-1, eps=1e-8)
            # normal to plane a-b-c
            n1 = F.normalize(torch.cross(ab, bc, dim=-1), dim=-1, eps=1e-8)
            # normal to plane b-c-d (will be rotated)
            # Create a vector perpendicular to bc in the plane defined by b,c,0
            # Standard method: create a reference vector that is not collinear with bc
            ref_vec = torch.tensor([1.0, 0.0, 0.0], device=device)
            if torch.abs(torch.dot(bc_norm, ref_vec)) > 0.9:
                ref_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
            perp = torch.cross(bc_norm, ref_vec, dim=-1)
            perp = F.normalize(perp, dim=-1, eps=1e-8)
            # This perp defines the zero dihedral direction
            # Now we need to rotate perp around bc_norm by (dihedral_ang0 + chi)
            chi_val = chi_angles[i, chi_idx]   # scalar
            total_angle = dihedral_ang0 + chi_val
            # Rodrigues rotation
            cos_a = torch.cos(total_angle)
            sin_a = torch.sin(total_angle)
            # rotated_perp = perp * cos_a + cross(bc_norm, perp) * sin_a
            cross_bn_perp = torch.cross(bc_norm, perp, dim=-1)
            rotated_perp = perp * cos_a + cross_bn_perp * sin_a
            # The bond direction is a combination of bc_norm (for angle) and rotated_perp
            # angle between bc and new bond is bond_ang
            # direction = cos(ang)*bc_norm + sin(ang)*rotated_perp
            bond_dir = math.cos(ang) * bc_norm + math.sin(ang) * rotated_perp
            new_pos = p_c_attach + bond_len * bond_dir

            local_atoms.append(new_pos)
            local_types.append(atom_type)
            chi_idx += 1

        res_atoms = torch.stack(local_atoms, dim=0)   # (M_i, 3)
        all_coords.append(res_atoms)
        all_types.append(local_types)

    return all_coords, all_types

def get_full_atom_coords_and_types(ca, seq, chi_angles):
    """Return a single tensor of all heavy atoms and types for the whole protein.
    Returns:
        coords: (N_total, 3)
        types: list of atom type strings length N_total
        residue_indices: (N_total,) tensor of residue index
    """
    res_coords, res_types = build_sidechain_atoms(ca, seq, chi_angles)
    coords_list = []
    types_list = []
    res_idx_list = []
    for i, (rc, rt) in enumerate(zip(res_coords, res_types)):
        coords_list.append(rc)
        types_list.extend(rt)
        res_idx_list.append(torch.full((rc.shape[0],), i, dtype=torch.long, device=ca.device))
    all_coords = torch.cat(coords_list, dim=0)
    res_indices = torch.cat(res_idx_list, dim=0)
    return all_coords, types_list, res_indices

# ──────────────────────────────────────────────────────────────────────────────
# Dihedral angle helpers (for backbone)
# ──────────────────────────────────────────────────────────────────────────────
def dihedral_angle(p0, p1, p2, p3):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = F.normalize(b1, dim=-1, eps=1e-8)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = torch.cross(b1n, v, dim=-1)
    y = (y * w).sum(-1)
    return torch.atan2(y + 1e-8, x + 1e-8)

def compute_phi_psi(atoms):
    N, CA, C = atoms['N'], atoms['CA'], atoms['C']
    L = CA.shape[0]
    phi = torch.zeros(L, device=CA.device)
    psi = torch.zeros(L, device=CA.device)
    if L > 2:
        phi[1:-1] = dihedral_angle(C[:-2], N[1:-1], CA[1:-1], C[1:-1])
        psi[1:-1] = dihedral_angle(N[1:-1], CA[1:-1], C[1:-1], N[2:])
    return phi * 180.0 / math.pi, psi * 180.0 / math.pi

# ──────────────────────────────────────────────────────────────────────────────
# Physics Energy Terms (backbone, from v26, adapted to use full-atom positions for LJ/Coulomb)
# ──────────────────────────────────────────────────────────────────────────────
def energy_bond(ca, alpha, cfg):
    target = cfg.ca_ca_dist * (1.0 + cfg.alpha_mod_bond * (alpha - 1.0))
    target_pair = 0.5 * (target[1:] + target[:-1])
    d = torch.norm(ca[1:] - ca[:-1], dim=-1)
    return cfg.w_bond * ((d - target_pair) ** 2).mean()

def energy_angle(ca, alpha, cfg):
    if len(ca) < 3:
        return torch.tensor(0.0, device=ca.device)
    v1 = ca[:-2] - ca[1:-1]
    v2 = ca[2:] - ca[1:-1]
    v1n = F.normalize(v1, dim=-1, eps=1e-8)
    v2n = F.normalize(v2, dim=-1, eps=1e-8)
    cos_ang = (v1n * v2n).sum(-1)
    target_angle = cfg.angle_target_rad * (1.0 + cfg.alpha_mod_angle * (alpha[1:-1] - 1.0))
    cos_target = torch.cos(target_angle)
    return cfg.w_angle * ((cos_ang - cos_target) ** 2).mean()

def energy_rama_vectorized(phi, psi, seq, alpha, cfg):
    L = len(seq)
    device = phi.device
    phi0 = torch.zeros(L, device=device)
    psi0 = torch.zeros(L, device=device)
    width = torch.zeros(L, device=device)
    for i, aa in enumerate(seq):
        prior = RAMACHANDRAN_PRIORS.get(aa, RAMACHANDRAN_PRIORS['general'])
        phi0[i] = prior['phi']
        psi0[i] = prior['psi']
        width[i] = prior['width']
    width_eff = width * (1.0 + cfg.alpha_mod_rama * (alpha - 1.0))
    dphi = (phi - phi0) / (width_eff + 1e-8)
    dpsi = (psi - psi0) / (width_eff + 1e-8)
    mask = torch.ones(L, device=device, dtype=torch.bool)
    mask[0] = False; mask[-1] = False
    loss = (dphi**2 + dpsi**2) * mask.float()
    return cfg.w_rama * loss.sum() / max(1, mask.sum())

def energy_hbond(atoms, alpha, cfg):
    O, N, C = atoms['O'], atoms['N'], atoms['C']
    D = torch.cdist(O, N)
    mask = (D > 2.5) & (D < 3.5)
    vec_co = O.unsqueeze(1) - C.unsqueeze(1)
    vec_no = N.unsqueeze(0) - O.unsqueeze(1)
    alignment = F.cosine_similarity(vec_co, vec_no, dim=-1, eps=1e-8)
    ideal_dist = 2.9 * (1.0 + cfg.alpha_mod_hbond * (alpha.unsqueeze(1) - 1.0))
    E = -alignment * torch.exp(-((D - ideal_dist) / 0.3) ** 2)
    return cfg.w_hbond * (E * mask.float()).mean()

def energy_electro(ca, seq, cfg):
    q = torch.tensor([RESIDUE_CHARGE.get(a, 0.0) for a in seq], device=ca.device)
    D = torch.cdist(ca, ca) + 1e-6
    E = q.unsqueeze(1) * q.unsqueeze(0) * torch.exp(-0.1 * D) / (80.0 * D)
    E.diagonal().zero_()
    return cfg.w_electro * E.mean()

def energy_solvent(ca, seq, cfg):
    D = torch.cdist(ca, ca)
    density = (D < 10.0).float().sum(dim=-1)
    burial = 1.0 - torch.exp(-density / 20.0)
    hydro = torch.tensor([HYDROPHOBICITY.get(a, 0.0) for a in seq], device=ca.device)
    exposed = torch.where(hydro > 0, hydro * (1.0 - burial), torch.zeros_like(burial))
    buried = torch.where(hydro <= 0, -hydro * burial, torch.zeros_like(burial))
    return cfg.w_solvent * (exposed + buried).mean()

# Backbone clash (CA only) we keep but also add full-atom LJ
def energy_clash(ca, alpha, cfg):
    D = torch.cdist(ca, ca)
    mask = torch.ones_like(D, dtype=torch.bool)
    idx = torch.arange(len(ca), device=ca.device)
    mask[idx[:, None], idx[None, :]] = False
    mask[idx[:-1, None], (idx[None, :-1]+1)] = False
    mask[(idx[None, :-1]+1), idx[:-1, None]] = False
    radius = cfg.clash_radius * (1.0 + cfg.alpha_mod_clash * (alpha.unsqueeze(1) - 1.0))
    radius_pair = 0.5 * (radius + radius.T)
    clash = torch.relu(radius_pair - D) * mask.float()
    return cfg.w_clash * (clash ** 2).mean()

# ──────────────────────────────────────────────────────────────────────────────
# Full‑Atom Energy: Lennard‑Jones (using sparse graph)
# ──────────────────────────────────────────────────────────────────────────────
def energy_lj_full(all_coords, all_types, res_indices, edge_index, edge_dist, cfg):
    """Lennard-Jones 12‑6 using pre‑built sparse graph on all heavy atoms.
    edge_index, edge_dist computed from all_coords.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=all_coords.device)
    src, dst = edge_index[0], edge_index[1]
    # Get LJ parameters for each atom
    sigmas = torch.zeros(len(all_types), device=all_coords.device)
    epsilons = torch.zeros(len(all_types), device=all_coords.device)
    for i, t in enumerate(all_types):
        s, e = LJ_PARAMS.get(t, (1.9, 0.1))
        sigmas[i] = s
        epsilons[i] = e
    sigma_src = sigmas[src]
    sigma_dst = sigmas[dst]
    eps_src = epsilons[src]
    eps_dst = epsilons[dst]
    sigma = 0.5 * (sigma_src + sigma_dst)
    eps = torch.sqrt(eps_src * eps_dst)
    # Exclude 1‑2, 1‑3 interactions based on residue indices and connectivity? We'll keep it simple: exclude within same residue? Actually, we exclude atoms that are covalently bonded or share a bond angle.
    # We'll use residue index and atom distance: if |res_i - res_j| <= 1 and they are not in the sparse graph due to cutoff? But we can approximate.
    # Since it's a refinement, clashes are important, we can simply allow all interactions but weight might be high.
    # We'll compute LJ:
    r = torch.clamp(edge_dist, min=1e-4)
    inv_r = 1.0 / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    lj = 4.0 * eps * ( (sigma * inv_r) ** 12 - (sigma * inv_r) ** 6 )
    return cfg.w_lj * lj.mean()

def energy_coulomb_full(all_coords, all_types, res_indices, edge_index, edge_dist, cfg):
    """Coulomb energy with distance‑dependent dielectric."""
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=all_coords.device)
    src, dst = edge_index[0], edge_index[1]
    # Partial charges: simple approximation based on atom type
    charge_map = {
        'N': -0.5, 'CA': 0.0, 'C': 0.5, 'O': -0.5, 'CB': 0.0,
        'CG': 0.0, 'CD': 0.0, 'CE': 0.0, 'CZ': 0.0,
        'OH': -0.5, 'OD': -0.5, 'OE': -0.5,
        'ND': -0.5, 'NE': -0.5, 'NH1': -0.5, 'NH2': -0.5,
        'SG': -0.2, 'S': -0.2, 'CH2': 0.0,
    }
    q = torch.tensor([charge_map.get(t, 0.0) for t in all_types], device=all_coords.device)
    qi = q[src]
    qj = q[dst]
    r = torch.clamp(edge_dist, min=1e-4)
    dielectric = 4.0 * r   # distance-dependent
    coulomb = 332.0637 * qi * qj / (dielectric * r)   # kcal/mol
    return cfg.w_coulomb * coulomb.mean()

def energy_torsion_chi(chi_angles, seq, cfg):
    """Simple torsional potential: penalize deviation from ideal chi values.
    chi_angles: (L, max_chi)
    We'll implement a multi‑well potential using cosine terms (like OPLS).
    For now, use a simple cosine that favors trans/gauche depending on residue.
    """
    L = len(seq)
    device = chi_angles.device
    max_chi = chi_angles.shape[1]
    energy = 0.0
    # For each residue, we have chi angles; define a simple potential: E = k * (1 + cos(n*chi - delta))
    # We'll use a rough approximation: for most residues chi1 prefers ~60,180,-60.
    # k = 1.0
    k = 1.0
    for i, aa in enumerate(seq):
        nchi = RESIDUE_NCHI.get(aa, 0)
        for c in range(min(nchi, max_chi)):
            chi = chi_angles[i, c]
            # simple: energy = 0.5 * (1 - cos(3*chi))  (period 120 deg)
            energy = energy + 0.5 * (1.0 - torch.cos(3.0 * chi))
    return cfg.w_torsion * energy / max(1, L)

# ──────────────────────────────────────────────────────────────────────────────
# SOC and Avalanche losses (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def sparse_soc_energy(ca, alpha, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=ca.device)
    src, dst = edge_index[0], edge_index[1]
    ai = alpha[src]
    aj = alpha[dst]
    a = 0.5 * (ai + aj)
    safe_dist = torch.clamp(edge_dist, min=1e-6)
    K = torch.exp(-a * torch.log(safe_dist)) * torch.exp(-edge_dist / cfg.kernel_lambda)
    E = -K * torch.exp(-edge_dist / 8.0)
    return cfg.w_soc_contact * E.mean()

def avalanche_loss_vec(coords, alpha, edge_index, edge_dist, cfg):
    if coords.grad is None:
        return torch.tensor(0.0, device=coords.device)
    L = coords.shape[0]
    stress = torch.norm(coords.grad, dim=-1)
    stressed = stress > cfg.avalanche_threshold
    if not stressed.any():
        return torch.tensor(0.0, device=coords.device)

    src = edge_index[0]
    dst = edge_index[1]
    ai = alpha[src]
    aj = alpha[dst]
    a = 0.5 * (ai + aj)
    safe_dist = torch.clamp(edge_dist, min=1e-6)
    K_edge = torch.exp(-a * torch.log(safe_dist)) * torch.exp(-edge_dist / cfg.kernel_lambda)

    direction = torch.zeros_like(coords)
    stressed_idx = torch.where(stressed)[0]
    grad_stressed = coords.grad[stressed_idx]
    norm = torch.norm(grad_stressed, dim=-1, keepdim=True)
    direction[stressed_idx] = -grad_stressed / (norm + 1e-8)

    src_stressed = stressed[src]
    if not src_stressed.any():
        return torch.tensor(0.0, device=coords.device)
    edge_K = K_edge[src_stressed]
    edge_dst = dst[src_stressed]
    edge_src = src[src_stressed]
    direction_src = direction[edge_src]
    coord_dst = coords[edge_dst]
    dot = (coord_dst * direction_src).sum(dim=-1)
    loss = - (edge_K * dot).mean()
    return cfg.w_avalanche * loss

def alpha_regularisation(alpha, cfg):
    entropy = -(alpha * torch.log(alpha + 1e-8)).mean()
    diff = alpha[1:] - alpha[:-1]
    smooth = (diff ** 2).mean()
    return cfg.w_alpha_entropy * entropy + cfg.w_alpha_smooth * smooth

# ──────────────────────────────────────────────────────────────────────────────
# Total Physics Energy (v27)
# ──────────────────────────────────────────────────────────────────────────────
def total_physics_energy_v27(ca, seq, alpha, chi_angles, edge_index_ca, edge_dist_ca, cfg):
    atoms = reconstruct_backbone(ca)   # N, CA, C, O
    phi, psi = compute_phi_psi(atoms)
    e = 0.0
    e += energy_bond(ca, alpha, cfg)
    e += energy_angle(ca, alpha, cfg)
    e += energy_rama_vectorized(phi, psi, seq, alpha, cfg)
    e += energy_clash(ca, alpha, cfg)
    e += energy_hbond(atoms, alpha, cfg)
    e += energy_electro(ca, seq, cfg)
    e += energy_solvent(ca, seq, cfg)
    # rotamer energy (CA‑based from v26) – we'll keep but it's superseded by full LJ; we can disable by setting w_rotamer=0
    e += cfg.w_rotamer * energy_rotamer_sparse(ca, atoms, seq, edge_index_ca, cfg)  # we need to define it, we'll reuse v26 version but with slight mods. We'll just include it with 0 weight if not needed.
    e += alpha_regularisation(alpha, cfg)

    # Full‑atom energies
    if chi_angles is not None:
        all_coords, all_types, res_indices = get_full_atom_coords_and_types(ca, seq, chi_angles)
        # Build sparse graph for full atom (can be heavy, use cutoff)
        edge_idx_full, edge_dist_full = build_sparse_graph(all_coords, cfg.sparse_cutoff)
        e += energy_lj_full(all_coords, all_types, res_indices, edge_idx_full, edge_dist_full, cfg)
        e += energy_coulomb_full(all_coords, all_types, res_indices, edge_idx_full, edge_dist_full, cfg)
        e += energy_torsion_chi(chi_angles, seq, cfg)

    if edge_index_ca is not None and edge_index_ca.numel() > 0:
        e += sparse_soc_energy(ca, alpha, edge_index_ca, edge_dist_ca, cfg)
        e += avalanche_loss_vec(ca, alpha, edge_index_ca, edge_dist_ca, cfg)
    return e

# We'll need the energy_rotamer_sparse function from v26, copied:
def energy_rotamer_sparse(ca, atoms, seq, edge_index, cfg):
    L = ca.shape[0]
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=ca.device)
    mask = torch.tensor([(aa != 'G' and i > 0 and i < L-1) for i, aa in enumerate(seq)],
                        device=ca.device)
    if not mask.any():
        return torch.tensor(0.0, device=ca.device)
    N = atoms['N']
    C = atoms['C']
    ca_m = ca[mask]
    n_m = N[mask]
    c_m = C[mask]
    v1 = n_m - ca_m
    v2 = c_m - ca_m
    cb_dir = -(v1 + v2)
    cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
    ideal_cb = ca_m + 1.8 * cb_dir
    global_idx = torch.where(mask)[0]
    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    is_masked_src = mask[edge_src]
    src_masked_idx = torch.where(is_masked_src)[0]
    src_global = edge_src[src_masked_idx]
    dst_global = edge_dst[src_masked_idx]
    global_to_masked = torch.full((L,), -1, device=ca.device, dtype=torch.long)
    global_to_masked[mask] = torch.arange(mask.sum(), device=ca.device)
    src_masked = global_to_masked[src_global]
    ca_dst = ca[dst_global]
    cb_src = ideal_cb[src_masked]
    dists = torch.norm(cb_src - ca_dst, dim=-1)
    not_self = (src_global != dst_global)
    dists = dists[not_self]
    src_masked = src_masked[not_self]
    min_per_masked = torch.full((mask.sum(),), float('inf'), device=ca.device)
    min_per_masked = torch.scatter_reduce(min_per_masked, 0, src_masked, dists, reduce='amin')
    min_per_masked[min_per_masked == float('inf')] = 10.0
    penalty = torch.relu(4.0 - min_per_masked)
    return cfg.w_rotamer * penalty.mean()

# ──────────────────────────────────────────────────────────────────────────────
# Dataset (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq, coords = self.data[idx]
        seq_ids = torch.tensor([AA_TO_ID.get(a,20) for a in seq], dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)
        return seq_ids, coords

def synthetic_dataset(num_samples=200, min_len=50, max_len=150):
    data = []
    for _ in range(num_samples):
        L = random.randint(min_len, max_len)
        seq = ''.join(random.choices(AA_VOCAB[:-1], k=L))
        coords = np.zeros((L,3), dtype=np.float32)
        d = np.random.randn(3).astype(np.float32)
        d /= np.linalg.norm(d)+1e-8
        for i in range(1,L):
            d += 0.2*np.random.randn(3).astype(np.float32)
            d /= np.linalg.norm(d)+1e-8
            coords[i] = coords[i-1] + d*3.8
        coords -= coords.mean(axis=0)
        data.append((seq, coords))
    return data

# ──────────────────────────────────────────────────────────────────────────────
# V27 Core Model
# ──────────────────────────────────────────────────────────────────────────────
class CSOCSSC_V27(nn.Module):
    def __init__(self, cfg: V27Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = FlashSequenceEncoder(cfg.dim, cfg.depth, cfg.heads, cfg.ff_mult)
        self.decoder = GeometryDecoder(cfg.dim)
        self.alpha_field = AdaptiveAlphaField(cfg.dim)
        self.csoc = CSOCController()
        self.rg = DiffRGRefiner(cfg.rg_factor) if cfg.use_rg else None

    def forward(self, seq_ids):
        latent = self.encoder(seq_ids)
        coords = self.decoder(latent)
        alpha = self.alpha_field(latent)
        return coords, alpha

    def predict(self, sequence):
        self.eval()
        with torch.no_grad():
            ids = torch.tensor([AA_TO_ID.get(a,20) for a in sequence],
                               dtype=torch.long, device=self.cfg.device).unsqueeze(0)
            coords, alpha = self.forward(ids)
        return coords.squeeze(0).cpu().numpy(), alpha.squeeze(0).cpu().numpy()

    def refine(self, sequence, init_coords=None, steps=None, logger=None):
        if steps is None:
            steps = self.cfg.refine_steps
        self.eval()
        device = torch.device(self.cfg.device)

        # Initial CA coords
        if init_coords is not None:
            init_centred = init_coords - init_coords.mean(axis=0)
            ca = torch.tensor(init_centred, dtype=torch.float32, device=device, requires_grad=True)
            with torch.no_grad():
                ids = torch.tensor([AA_TO_ID.get(a,20) for a in sequence],
                                   dtype=torch.long, device=device).unsqueeze(0)
                latent = self.encoder(ids)
                alpha = self.alpha_field(latent).squeeze(0)
        else:
            with torch.no_grad():
                coords_np, alpha_np = self.predict(sequence)
            ca = torch.tensor(coords_np, dtype=torch.float32, device=device, requires_grad=True)
            alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

        L = len(sequence)

        # Initialise chi angles as optimizable parameters
        max_chi = MAX_CHI
        chi_init = torch.zeros((L, max_chi), device=device)
        # Random starting chi angles (uniform)
        chi_init.uniform_(-math.pi, math.pi)
        chi = nn.Parameter(chi_init)

        # Build sparse graph for CA
        edge_index_ca, edge_dist_ca = build_sparse_graph(ca, self.cfg.sparse_cutoff)

        # Neural restraint (CA only)
        neural_target = None
        if init_coords is None:
            with torch.no_grad():
                neural_coords_np, _ = self.predict(sequence)
                neural_target = torch.tensor(neural_coords_np, device=device)

        opt = torch.optim.Adam([ca, chi], lr=self.cfg.lr)
        scaler = GradScaler(enabled=self.cfg.use_amp)

        energy_history = []
        for step in range(steps):
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=self.cfg.use_amp):
                e_phys = total_physics_energy_v27(ca, sequence, alpha, chi,
                                                  edge_index_ca, edge_dist_ca, self.cfg)
                loss = e_phys
                if neural_target is not None:
                    loss = loss + 0.1 * ((ca - neural_target) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([ca, chi], max_norm=10.0)
            scaler.step(opt)
            scaler.update()

            sigma = self.csoc.sigma(ca.detach())
            T = self.csoc.temperature(sigma, self.cfg.temp_base, self.cfg.sigma_target)
            noise_scale = math.sqrt(2 * self.cfg.friction * T.item() / 300.0) * self.cfg.lr
            with torch.no_grad():
                ca.add_(torch.randn_like(ca) * noise_scale)
                # Also add small noise to chi
                chi_noise = torch.randn_like(chi) * noise_scale * 0.5
                chi.data.add_(chi_noise)

            if step > 0 and step % self.cfg.rebuild_interval == 0:
                edge_index_ca, edge_dist_ca = build_sparse_graph(ca.detach(), self.cfg.sparse_cutoff)

            if self.rg is not None and step > 0 and step % self.cfg.rg_interval == 0:
                ca.data = self.rg.forward(ca.data)

            if step % 50 == 0 and logger:
                logger.info(f"refine {step:04d}  loss={loss.item():.4f}  phys={e_phys.item():.4f}  "
                            f"σ={sigma.item():.3f}  T={T.item():.1f}")
                energy_history.append(loss.item())

        # Return CA coords and final chi for full atom output
        return ca.detach().cpu().numpy(), chi.detach().cpu().numpy(), energy_history

# ──────────────────────────────────────────────────────────────────────────────
# Training (unchanged, CA‑level)
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model, dataloader, cfg, logger):
    device = torch.device(cfg.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.use_amp)

    for epoch in range(cfg.epochs):
        if cfg.is_distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        total_loss = 0.0
        optimizer.zero_grad()
        for step, (seq_ids, target_coords) in enumerate(dataloader):
            seq_ids, target_coords = seq_ids.to(device), target_coords.to(device)
            with autocast(device_type=device.type, enabled=cfg.use_amp):
                pred_coords, pred_alpha = model(seq_ids)
                coord_loss = F.mse_loss(pred_coords, target_coords)
                alpha_reg = 0.001 * ((pred_alpha[:,1:] - pred_alpha[:,:-1])**2).mean()
                loss = (coord_loss + alpha_reg) / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * cfg.gradient_accumulation_steps

        if cfg.local_rank in [-1, 0]:
            logger.info(f"Epoch {epoch+1:03d}/{cfg.epochs}  MSE={total_loss/len(dataloader):.4f}")

    if cfg.local_rank in [-1, 0]:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(cfg.checkpoint_dir, "v27_pretrained.pt")
        state_dict = model.module.state_dict() if cfg.is_distributed else model.state_dict()
        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def compute_rmsd(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return float(np.sqrt(np.mean(np.sum((a @ R - b)**2, axis=1))))

def write_full_pdb(ca, seq, chi_angles, filename):
    """Write full heavy‑atom PDB from refined CA and chi angles."""
    device = 'cpu'
    ca_t = torch.tensor(ca, dtype=torch.float32)
    chi_t = torch.tensor(chi_angles, dtype=torch.float32)
    all_coords, all_types, res_indices = get_full_atom_coords_and_types(ca_t, seq, chi_t)
    # Also need O backbone atom; we'll rebuild backbone and add O.
    atoms = reconstruct_backbone(ca_t)  # gives N, CA, C, O
    N_coords = atoms['N']
    C_coords = atoms['C']
    O_coords = atoms['O']
    # Build a list of atoms with proper names per residue.
    with open(filename, 'w') as f:
        atom_serial = 1
        for i, (aa, ca_i) in enumerate(zip(seq, ca)):
            # Write backbone N, CA, C, O
            f.write(f"ATOM  {atom_serial:5d}  N   {aa:3s} A{i+1:4d}    "
                    f"{N_coords[i,0]:8.3f}{N_coords[i,1]:8.3f}{N_coords[i,2]:8.3f}  1.00  0.00           N\n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:5d}  CA  {aa:3s} A{i+1:4d}    "
                    f"{ca_i[0]:8.3f}{ca_i[1]:8.3f}{ca_i[2]:8.3f}  1.00  0.00           C\n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:5d}  C   {aa:3s} A{i+1:4d}    "
                    f"{C_coords[i,0]:8.3f}{C_coords[i,1]:8.3f}{C_coords[i,2]:8.3f}  1.00  0.00           C\n")
            atom_serial += 1
            f.write(f"ATOM  {atom_serial:5d}  O   {aa:3s} A{i+1:4d}    "
                    f"{O_coords[i,0]:8.3f}{O_coords[i,1]:8.3f}{O_coords[i,2]:8.3f}  1.00  0.00           O\n")
            atom_serial += 1

            # Write sidechain atoms from all_coords (excluding N, CA, C which are already done)
            # We need to extract atoms for this residue.
            # all_coords is concatenated per residue, we can use res_indices to filter.
            res_atom_mask = res_indices == i
            res_coords = all_coords[res_atom_mask]
            res_types = [all_types[j] for j in range(len(all_types)) if res_indices[j]==i]
            # The builder returns N, CA, C, CB, ... for non‑Gly, and N, CA, C for Gly.
            # So we need to skip N, CA, C.
            if aa != 'G':
                # first three are N, CA, C (already written), then CB onwards
                # indices: 0:N,1:CA,2:C,3:CB,...
                for k in range(3, res_coords.shape[0]):
                    at_name = res_types[k]
                    x,y,z = res_coords[k]
                    f.write(f"ATOM  {atom_serial:5d}  {at_name:4s}{aa:3s} A{i+1:4d}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {at_name[0]}\n")
                    atom_serial += 1
            else:
                # Gly: no sidechain, already wrote N,CA,C,O
                pass
        f.write("END\n")
    print(f"Full‑atom PDB written to {filename}")

# ──────────────────────────────────────────────────────────────────────────────
# Main CLI (adapted)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSOC‑SSC V27 Full‑Atom Folding Engine")
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train')
    train_parser.add_argument('--samples', type=int, default=1000)
    train_parser.add_argument('--epochs', type=int, default=80)
    train_parser.add_argument('--batch_size', type=int, default=8)

    refine_parser = sub.add_parser('refine')
    refine_parser.add_argument('--seq', type=str, default=None)
    refine_parser.add_argument('--pdb', type=str, default=None)
    refine_parser.add_argument('--init', type=str, default=None)
    refine_parser.add_argument('--out', type=str, default='refined_v27.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--checkpoint', type=str, default='v27_pretrained.pt')

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = V27Config(local_rank=local_rank, is_distributed=is_distributed, device=device_str,
                    epochs=getattr(args,'epochs',80), batch_size=getattr(args,'batch_size',8),
                    refine_steps=getattr(args,'steps',600))

    torch.manual_seed(cfg.seed + (local_rank if local_rank>0 else 0))
    np.random.seed(cfg.seed + (local_rank if local_rank>0 else 0))
    random.seed(cfg.seed + (local_rank if local_rank>0 else 0))

    logger = setup_logger("CSOC-SSC_V27", local_rank)

    if local_rank in [-1,0]:
        logger.info("="*60)
        logger.info("CSOC-SSC V27 – Full‑Atom GPU‑Native SOC Folding Engine")
        logger.info(f"Distributed: {is_distributed} | Device: {device_str}")
        logger.info("="*60)

    if args.command == 'train':
        if local_rank in [-1,0]:
            logger.info("Generating synthetic training data...")
        data = synthetic_dataset(num_samples=args.samples)
        dataset = ProteinDataset(data)
        if is_distributed:
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        model = CSOCSSC_V27(cfg).to(torch.device(device_str))
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        train_model(model, dataloader, cfg, logger)

    elif args.command == 'refine':
        model = CSOCSSC_V27(cfg).to(torch.device(device_str))
        ckpt = args.checkpoint
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device_str))
            logger.info(f"Loaded weights from {ckpt}")
        else:
            logger.warning("Checkpoint not found; using random weights.")

        target_seq = args.seq
        init_coords = None
        native_coords = None
        if args.pdb:
            logger.info(f"Fetching PDB {args.pdb}...")
            backbone = PDBFetcher.fetch_and_parse(args.pdb)
            init_coords = backbone.ca
            target_seq = backbone.seq
            native_coords = backbone.native_coords
        elif args.init and os.path.exists(args.init):
            coords_list = []
            with open(args.init) as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip()=='CA':
                        coords_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if coords_list:
                init_coords = np.array(coords_list, dtype=np.float32)
                logger.info(f"Loaded {len(init_coords)} initial CA atoms.")
        if not target_seq:
            raise ValueError("Must provide --seq or --pdb.")

        start_time = time.time()
        refined_ca, refined_chi, _ = model.refine(target_seq, init_coords=init_coords,
                                                   steps=cfg.refine_steps, logger=logger)
        write_full_pdb(refined_ca, target_seq, refined_chi, args.out)
        logger.info(f"Full‑atom refined structure saved to {args.out}")
        if native_coords is not None:
            rmsd_val = compute_rmsd(refined_ca, native_coords)
            logger.info(f"Final CA RMSD vs Native: {rmsd_val:.4f} Å")
        logger.info(f"Compute Time: {time.time()-start_time:.2f} seconds")

    if is_distributed:
        dist.destroy_process_group()

#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC v30.1.1.1 — MSA‑Enabled Hybrid Folding Engine
# =============================================================================
import os, math, time, random, argparse, logging, json, glob, urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch_cluster import radius_graph, radius
import warnings
warnings.filterwarnings("ignore")

# ──────────────── Logging ────────────────
def setup_logger(name="CSOC‑SSC_V30.1.1.1", local_rank=-1):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('[%(asctime)s] [Rank %(process)d] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(h)
    logger.setLevel(logging.INFO if local_rank in [-1, 0] else logging.WARNING)
    return logger

# ──────────────── Constants ────────────────
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_VOCAB)}
HYDROPHOBICITY = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
    'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,
    'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3,'X':0.0
}
RESIDUE_CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.5}
RAMACHANDRAN_PRIORS = {
    'general':{'phi':-60.0,'psi':-45.0,'width':25.0},
    'G':{'phi':-75.0,'psi':-60.0,'width':40.0},
    'P':{'phi':-65.0,'psi':-30.0,'width':20.0},
}
AA_3_TO_1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
             'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
             'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

RESIDUE_TOPOLOGY = {
    'G':[],
    'A':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0)],
    'S':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('OG','OH',1,1.43,109.5,(-2,-1,0),0.0)],
    'C':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('SG','SG',1,1.81,109.5,(-2,-1,0),0.0)],
    'V':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG1','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0)],
    'T':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('OG1','OH',1,1.43,109.5,(-2,-1,0),0.0),
         ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0)],
    'L':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CD1','CB',2,1.53,109.5,(-1,0,1),0.0),
         ('CD2','CB',2,1.53,109.5,(-1,0,1),2.0)],
    'I':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG1','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0),
         ('CD1','CB',2,1.53,109.5,(-1,0,1),0.0)],
    'M':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('SD','S',2,1.81,109.5,(-1,0,1),0.0),
         ('CE','CB',3,1.81,109.5,(-2,-1,0),0.0)],
    'F':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0),
         ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0),
         ('CE1','CB',3,1.40,120.0,(2,1,0),0.0),
         ('CE2','CB',4,1.40,120.0,(2,1,0),2.0),
         ('CZ','CB',5,1.40,120.0,(3,2,1),0.0)],
    'Y':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0),
         ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0),
         ('CE1','CB',3,1.40,120.0,(2,1,0),0.0),
         ('CE2','CB',4,1.40,120.0,(2,1,0),2.0),
         ('CZ','CB',5,1.40,120.0,(3,2,1),0.0),
         ('OH','OH',6,1.36,120.0,(4,3,2),0.0)],
    'W':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0),
         ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0),
         ('NE1','N',3,1.38,120.0,(2,1,0),0.0),
         ('CE2','CB',4,1.40,120.0,(2,1,0),2.0),
         ('CE3','CB',5,1.40,120.0,(2,1,0),2.0),
         ('CZ2','CB',6,1.40,120.0,(3,2,1),0.0),
         ('CZ3','CB',7,1.40,120.0,(5,4,2),0.0),
         ('CH2','CB',8,1.40,120.0,(6,3,2),0.0)],
    'D':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','C',1,1.52,109.5,(-2,-1,0),0.0),
         ('OD1','O',2,1.25,120.0,(-1,0,1),0.0),
         ('OD2','O',2,1.25,120.0,(-1,0,1),2.0)],
    'E':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.52,109.5,(-2,-1,0),0.0),
         ('CD','C',2,1.52,109.5,(-1,0,1),0.0),
         ('OE1','O',3,1.25,120.0,(2,1,0),0.0),
         ('OE2','O',3,1.25,120.0,(2,1,0),2.0)],
    'N':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','C',1,1.52,109.5,(-2,-1,0),0.0),
         ('OD1','O',2,1.25,120.0,(-1,0,1),0.0),
         ('ND2','N',2,1.33,120.0,(-1,0,1),2.0)],
    'Q':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.52,109.5,(-2,-1,0),0.0),
         ('CD','C',2,1.52,109.5,(-1,0,1),0.0),
         ('OE1','O',3,1.25,120.0,(2,1,0),0.0),
         ('NE2','N',3,1.33,120.0,(2,1,0),2.0)],
    'K':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.52,109.5,(-2,-1,0),0.0),
         ('CD','CB',2,1.52,109.5,(-1,0,1),0.0),
         ('CE','CB',3,1.52,109.5,(-2,-1,0),0.0),
         ('NZ','N',4,1.47,109.5,(-1,0,1),0.0)],
    'R':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.52,109.5,(-2,-1,0),0.0),
         ('CD','CB',2,1.52,109.5,(-1,0,1),0.0),
         ('NE','N',3,1.46,109.5,(-2,-1,0),0.0),
         ('CZ','C',4,1.33,125.0,(-1,0,1),0.0),
         ('NH1','N',5,1.33,120.0,(4,3,2),0.0),
         ('NH2','N',5,1.33,120.0,(4,3,2),2.0)],
    'H':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.50,109.5,(-2,-1,0),0.0),
         ('ND1','N',2,1.38,120.0,(-1,0,1),0.0),
         ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0),
         ('CE1','CB',3,1.40,120.0,(2,1,0),0.0),
         ('NE2','N',4,1.38,120.0,(2,1,0),2.0)],
    'P':[('CB','CB',0,1.53,104.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.50,104.5,(-2,-1,0),0.0),
         ('CD','CB',2,1.50,104.5,(-1,0,1),0.0)],
}
MAX_CHI = 4
RESIDUE_NCHI = {
    'A':0,'G':0,'S':1,'C':1,'V':1,'T':1,'L':2,'I':2,'M':3,'F':2,'Y':2,'W':2,
    'D':2,'E':3,'N':2,'Q':3,'K':4,'R':4,'H':2,'P':2
}

DEFAULT_LJ_PARAMS = {
    'C':(1.9080,0.0860),'CA':(1.9080,0.0860),'CB':(1.9080,0.0860),
    'CG':(1.9080,0.0860),'CD':(1.9080,0.0860),'CE':(1.9080,0.0860),
    'CZ':(1.9080,0.0860),'CH2':(1.9080,0.0860),
    'N':(1.8240,0.1700),'ND':(1.8240,0.1700),'NE':(1.8240,0.1700),
    'NH1':(1.8240,0.1700),'NH2':(1.8240,0.1700),
    'O':(1.6612,0.2100),'OD':(1.6612,0.2100),'OE':(1.6612,0.2100),
    'OH':(1.6612,0.2100),'S':(2.0000,0.2500),'SG':(2.0000,0.2500),
}
DEFAULT_CHARGE_MAP = {'N':-0.5,'CA':0.0,'C':0.5,'O':-0.5,'CB':0.0,'OH':-0.5,
                      'OD':-0.5,'OE':-0.5,'ND':-0.5,'NE':-0.5,'NH1':-0.5,
                      'NH2':-0.5,'SG':-0.2,'S':-0.2}

def load_forcefield_params(lj_file=None, charge_file=None):
    lj = DEFAULT_LJ_PARAMS.copy()
    charges = DEFAULT_CHARGE_MAP.copy()
    if lj_file and os.path.exists(lj_file):
        with open(lj_file,'r') as f:
            data = json.load(f)
            for k,v in data.items():
                if isinstance(v,list) and len(v)==2: lj[k] = tuple(v)
    if charge_file and os.path.exists(charge_file):
        with open(charge_file,'r') as f: charges.update(json.load(f))
    return lj, charges

# ──────────────── DNA/RNA Constants (embedded) ────────────────
DNA_RNA_VOCAB = "ACGTU"
NT_TO_ID = {nt:i for i,nt in enumerate(DNA_RNA_VOCAB)}

WC_PAIRS = {
    ('A','T'):2,('T','A'):2,('A','U'):2,('U','A'):2,
    ('G','C'):3,('C','G'):3,('G','U'):1,('U','G'):1
}
BASE_STACKING = {'A':1.0,'T':0.8,'U':0.8,'G':1.2,'C':1.0}

NUCLEOTIDE_LJ = {
    'P':(2.1000,0.2000),'O':(1.6612,0.2100),'N':(1.8240,0.1700),
    'C':(1.9080,0.0860),'S':(2.0000,0.2500)
}
NUCLEOTIDE_CHARGES = {
    'P':0.90,'OP1':-0.70,'OP2':-0.70,'O5\'':-0.50,'C5\'':-0.10,'C4\'':0.00,'O4\'':-0.40,'C1\'':0.20,'C2\'':-0.10,'C3\'':0.10,'O3\'':-0.50,
    'N1':-0.50,'N2':-0.80,'N3':-0.60,'N4':-0.80,'N6':-0.80,'N7':-0.50,'N9':-0.30,
    'C2':0.40,'C4':0.30,'C5':0.10,'C6':0.10,'C7':-0.20,'C8':0.20,
    'O2':-0.55,'O4':-0.55,'O6':-0.55,
}

def detect_sequence_type(seq: str) -> str:
    """Auto‑detect if sequence is protein, DNA, or RNA."""
    nt_set = set("ACGTU")
    aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    n_nt = sum(1 for c in seq.upper() if c in nt_set)
    n_aa = sum(1 for c in seq.upper() if c in aa_set)
    total = len(seq)
    if total == 0: return 'unknown'
    if n_nt/total > 0.8:
        if 'U' in seq.upper(): return 'rna'
        return 'dna'
    if n_aa/total > 0.7: return 'protein'
    return 'unknown'

# ═══════════════════════════════════════════════════════════════
# DNA/RNA Energy Functions (lightweight, no import needed)
# ═══════════════════════════════════════════════════════════════
def build_dna_rna_c4_bond_energy(c4, w=30.0, ideal=6.5):
    if c4.shape[0]<2: return torch.tensor(0.0, device=c4.device)
    d = torch.norm(c4[1:]-c4[:-1], dim=-1)
    return w*((d-ideal)**2).mean()

def dna_rna_base_pair_energy(c4, seq, w=8.0, ideal=10.5, sigma=2.0):
    L = len(seq); device=c4.device; energy=torch.tensor(0.0,device=device)
    for i in range(L):
        for j in range(i+4, min(L, i+50)):
            d = torch.norm(c4[i]-c4[j])
            pair = (seq[i], seq[j])
            nb = WC_PAIRS.get(pair, 0)
            if nb>0:
                energy += -nb * torch.exp(-((d-ideal)/sigma)**2)
    return w*energy/max(1,L)

def dna_rna_stacking_energy(c4, seq, w=5.0, ideal=6.5, sigma=1.5):
    L = len(seq); device=c4.device; energy=torch.tensor(0.0,device=device)
    for i in range(L-1):
        d = torch.norm(c4[i+1]-c4[i])
        si = BASE_STACKING.get(seq[i],1.0)
        sj = BASE_STACKING.get(seq[i+1],1.0)
        s_avg = 0.5*(si+sj)
        energy += -s_avg * torch.exp(-((d-ideal)/sigma)**2)
    return w*energy/max(1,L-1)

def dna_rna_lj_energy(all_coords, all_types, edge_index, edge_dist, w=30.0, lj_params=None):
    if edge_index.numel()==0: return torch.tensor(0.0, device=all_coords.device)
    if lj_params is None: lj_params = NUCLEOTIDE_LJ
    src,dst = edge_index
    # map to element type (simple)
    elem = [t[0] if t[0] in 'CNOPS' else 'C' for t in all_types]
    sigmas = torch.tensor([lj_params.get(e,(1.9,0.1))[0] for e in elem], device=all_coords.device)
    epsilons = torch.tensor([lj_params.get(e,(1.9,0.1))[1] for e in elem], device=all_coords.device)
    sig_ij = 0.5*(sigmas[src]+sigmas[dst])
    eps_ij = torch.sqrt(epsilons[src]*epsilons[dst])
    r = torch.clamp(edge_dist, min=1e-4)
    inv_r = 1.0/r; inv_r6=inv_r**6; inv_r12=inv_r6**2
    return w*(4.0*eps_ij*( (sig_ij*inv_r)**12 - (sig_ij*inv_r)**6 )).mean()

def dna_rna_coulomb_energy(all_coords, all_types, edge_index, edge_dist, w=3.0, charge_map=None):
    if edge_index.numel()==0: return torch.tensor(0.0, device=all_coords.device)
    if charge_map is None: charge_map = NUCLEOTIDE_CHARGES
    src,dst = edge_index
    q = torch.tensor([charge_map.get(t,0.0) for t in all_types], device=all_coords.device)
    qi,qj = q[src], q[dst]
    r = torch.clamp(edge_dist, min=1e-4)
    dielectric = 4.0 * r
    return w*(332.0637*qi*qj/(dielectric*r+1e-8)).mean()

def build_dna_rna_full_atoms(c4, seq):
    # Simplified placeholder that returns c4 only; full atom builder can be added if needed.
    # For now we skip full atom building and just use c4 for LJ/Coulomb on C4' only.
    all_coords = c4
    all_types = ['C4\'']*len(c4)  # treat as carbon
    res_idx = torch.arange(len(c4), device=c4.device)
    return all_coords, all_types, res_idx

class DNA_RNA_Energy:
    def __init__(self, w_c4_bond=30.0, w_bp=8.0, w_stack=5.0, w_lj=30.0, w_coulomb=3.0):
        self.w_c4_bond=w_c4_bond; self.w_bp=w_bp; self.w_stack=w_stack
        self.w_lj=w_lj; self.w_coulomb=w_coulomb
    def __call__(self, c4, seq, edge_index=None, edge_dist=None):
        E = build_dna_rna_c4_bond_energy(c4, self.w_c4_bond)
        E += dna_rna_base_pair_energy(c4, seq, self.w_bp)
        E += dna_rna_stacking_energy(c4, seq, self.w_stack)
        if edge_index is not None and edge_index.numel()>0:
            all_c, all_t, _ = build_dna_rna_full_atoms(c4, seq)
            E += dna_rna_lj_energy(all_c, all_t, edge_index, edge_dist, self.w_lj)
            E += dna_rna_coulomb_energy(all_c, all_t, edge_index, edge_dist, self.w_coulomb)
        return E

# ═══════════════════════════════════════════════════════════════
# Configuration V30.1.1.1
# ═══════════════════════════════════════════════════════════════
@dataclass
class V30_1_1Config:
    local_rank: int = int(os.environ.get("LOCAL_RANK", -1))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    dim: int = 256
    depth: int = 6
    heads: int = 8
    ff_mult: int = 4

    use_msa: bool = False
    msa_n_seq: int = 128
    msa_dim: int = 64

    egnn_layers: int = 4
    egnn_hidden: int = 128
    egnn_edge_dim: int = 32
    egnn_cutoff: float = 15.0

    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 80
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    pdb_dir: str = "./pdbs"
    msa_dir: str = "./msas"

    refine_steps: int = 600
    temp_base: float = 300.0
    friction: float = 0.02
    sigma_target: float = 1.0
    avalanche_threshold: float = 0.5
    w_avalanche: float = 0.2

    w_bond: float = 30.0
    w_angle: float = 15.0
    w_rama: float = 8.0
    w_clash: float = 80.0
    w_hbond: float = 6.0
    w_electro: float = 4.0
    w_solvent: float = 5.0
    w_rotamer: float = 3.0
    w_lj: float = 50.0
    w_coulomb: float = 5.0
    w_torsion: float = 10.0

    w_soc_contact: float = 0.3
    kernel_lambda: float = 12.0
    rebuild_interval: int = 100

    alpha_mod_bond: float = 0.1
    alpha_mod_angle: float = 0.05
    alpha_mod_rama: float = 0.2
    alpha_mod_clash: float = 0.1
    alpha_mod_hbond: float = 0.1
    w_alpha_entropy: float = 0.5
    w_alpha_smooth: float = 0.1

    chain_break_weight: float = 1.0
    sparse_cutoff: float = 12.0
    max_neighbors: int = 64
    lj_param_file: str = None
    charge_param_file: str = None

    use_pme: bool = True
    pme_grid: int = 64
    pme_alpha: float = 0.25

    use_torch_compile: bool = False  # safer default
    compile_mode: str = 'max-autotune'
    zero_copy_pinned: bool = True
    num_replicas: int = 1

    use_rg: bool = True
    rg_factor: int = 4
    rg_interval: int = 200
    checkpoint_dir: str = "./v30_1_1_ckpt"
    out_pdb: str = "refined_v30_1_1.pdb"

    # DNA/RNA bridge weights
    w_dna_c4_bond: float = 30.0
    w_dna_bp: float = 8.0
    w_dna_stack: float = 5.0
    w_dna_lj: float = 30.0
    w_dna_coulomb: float = 3.0

    def __post_init__(self):
        self.lj_params, self.charge_map = load_forcefield_params(self.lj_param_file, self.charge_param_file)
        self.clash_radius = 2.0
        self.angle_target_rad = 111.0 * math.pi / 180.0
        self.dna_rna_eng = DNA_RNA_Energy(w_c4_bond=self.w_dna_c4_bond, w_bp=self.w_dna_bp,
                                          w_stack=self.w_dna_stack, w_lj=self.w_dna_lj, w_coulomb=self.w_dna_coulomb)

# ──────────────── Datasets ────────────────
@dataclass
class Backbone:
    ca: np.ndarray
    seq: str
    chain_id: str = ""
    native_coords: Optional[np.ndarray] = None

class MultimerPDBFetcher:
    @staticmethod
    def fetch(pdb_id: str) -> Tuple[List[Backbone], List[str]]:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        req = urllib.request.Request(url, headers={'User-Agent': 'CSOC-SSC_V30.1.1.1'})
        chains_data = {}
        chain_order = []
        try:
            with urllib.request.urlopen(req) as response:
                lines = response.read().decode('utf-8').split('\n')
            for line in lines:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    chain = line[21].strip()
                    res_name = line[17:20].strip()
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    if chain not in chains_data:
                        chains_data[chain] = {'res':[], 'coords':[]}
                        chain_order.append(chain)
                    chains_data[chain]['res'].append(res_name)
                    chains_data[chain]['coords'].append([x,y,z])
        except Exception as e:
            raise RuntimeError(f"Failed to fetch PDB {pdb_id}: {e}")
        backbones = []
        for chain in chain_order:
            seq = "".join([AA_3_TO_1.get(r,'X') for r in chains_data[chain]['res']])
            coords = np.array(chains_data[chain]['coords'], dtype=np.float32)
            rand_coords = coords + np.random.randn(*coords.shape)*10.0
            backbones.append(Backbone(ca=rand_coords, seq=seq, chain_id=chain, native_coords=coords))
        return backbones, chain_order

    @staticmethod
    def fetch_from_file(filepath):
        chains_data = {}
        chain_order = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip()=='CA':
                    chain = line[21].strip()
                    res_name = line[17:20].strip()
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    if chain not in chains_data:
                        chains_data[chain] = {'res':[], 'coords':[]}
                        chain_order.append(chain)
                    chains_data[chain]['res'].append(res_name)
                    chains_data[chain]['coords'].append([x,y,z])
        backbones = []
        for chain in chain_order:
            seq = "".join([AA_3_TO_1.get(r,'X') for r in chains_data[chain]['res']])
            coords = np.array(chains_data[chain]['coords'], dtype=np.float32)
            backbones.append(Backbone(ca=coords, seq=seq, chain_id=chain, native_coords=coords))
        return backbones, chain_order

class RealProteinDataset(Dataset):
    def __init__(self, pdb_dir, max_len=500):
        self.samples = []
        for fpath in glob.glob(os.path.join(pdb_dir, "*.pdb")):
            try:
                backbones, _ = MultimerPDBFetcher.fetch_from_file(fpath)
                for bb in backbones:
                    if 10 <= len(bb.seq) <= max_len:
                        self.samples.append((bb.seq, bb.ca))
            except Exception: pass
        if not self.samples:
            raise RuntimeError(f"No valid protein chains found in {pdb_dir}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, coords = self.samples[idx]
        seq_ids = torch.tensor([AA_TO_ID.get(a,20) for a in seq], dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)
        return seq_ids, coords

class MSAProteinDataset(Dataset):
    def __init__(self, pdb_dir, msa_dir, max_len=500, max_n_seq=128):
        self.samples = []
        pdb_files = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(pdb_dir, "*.pdb"))}
        for msa_path in glob.glob(os.path.join(msa_dir, "*.npy")):
            pdb_id = os.path.splitext(os.path.basename(msa_path))[0]
            if pdb_id not in pdb_files: continue
            try:
                backbones, _ = MultimerPDBFetcher.fetch_from_file(pdb_files[pdb_id])
                for bb in backbones:
                    if 10 <= len(bb.seq) <= max_len:
                        msa = np.load(msa_path)
                        if msa.shape[1] != len(bb.seq): continue
                        msa = msa[:max_n_seq] if msa.shape[0] > max_n_seq else msa
                        self.samples.append((bb.seq, bb.ca, msa))
            except Exception: pass
        if not self.samples:
            raise RuntimeError(f"No MSA–PDB pairs found in {msa_dir} and {pdb_dir}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, coords, msa = self.samples[idx]
        seq_ids = torch.tensor([AA_TO_ID.get(a,20) for a in seq], dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)
        msa = torch.tensor(msa, dtype=torch.float32)
        return seq_ids, coords, msa

# ──────────────── MSA Encoder ────────────────
class AxialAttention(nn.Module):
    def __init__(self, dim, heads, axis='row'):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.axis = axis
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, N_seq, L, C = x.shape
        if self.axis == 'row':
            x_reshape = x.permute(0,2,1,3).reshape(B*L, N_seq, C)
        else:
            x_reshape = x.reshape(B*N_seq, L, C)
        residual = x_reshape
        qkv = self.qkv(x_reshape).reshape(-1, x_reshape.size(1), 3, self.heads, self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
        attn_out = attn_out.permute(0,2,1,3).reshape_as(residual)
        attn_out = self.proj(attn_out)
        out = self.norm(residual + self.dropout(attn_out))
        if self.axis == 'row':
            out = out.reshape(B, L, N_seq, C).permute(0,2,1,3)
        else:
            out = out.reshape(B, N_seq, L, C)
        return out

class MSAEncoder(nn.Module):
    def __init__(self, msa_dim, n_layers=4, heads=4, out_dim=256):
        super().__init__()
        self.embed = nn.Linear(22, msa_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([AxialAttention(msa_dim, heads, 'row'),
                           AxialAttention(msa_dim, heads, 'column')])
            for _ in range(n_layers)
        ])
        self.final_linear = nn.Linear(msa_dim, out_dim)

    def forward(self, msa):
        x = self.embed(msa)
        for row_attn, col_attn in self.layers:
            x = row_attn(x)
            x = col_attn(x)
        x = x.mean(dim=1)
        return self.final_linear(x)

# ──────────────── Single‑Seq Encoder ────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0)/dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class FlashGeometryBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*ff_mult), nn.GELU(), nn.Linear(dim*ff_mult, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attn_mask=None):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(q,k,v, attn_mask=attn_mask, dropout_p=0.1 if self.training else 0.0)
        attn_out = attn_out.permute(0,2,1,3).reshape(B,N,C)
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

class AdaptiveAlphaField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
    def forward(self, latent):
        a = torch.sigmoid(self.net(latent))
        a = 0.5 + 2.5 * a.squeeze(-1)
        return torch.clamp(a, 0.5, 3.0)

# ──────────────── EGNN Decoder ────────────────
class EGNNLayer(nn.Module):
    def __init__(self, node_dim, hidden_dim, edge_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(nn.Linear(node_dim*2+edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_dim))
        self.coord_mlp = nn.Sequential(nn.Linear(node_dim*2+edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim,1,bias=False))
        self.edge_mlp = nn.Sequential(nn.Linear(1, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim))

    def forward(self, h, x, edge_index, edge_dist):
        src, dst = edge_index
        edge_attr = self.edge_mlp(edge_dist.unsqueeze(-1))
        m_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        m = self.node_mlp(m_input)
        h_agg = torch.zeros_like(h)
        h_agg = h_agg.index_add(0, dst, m)
        coord_weight = self.coord_mlp(m_input)
        dir_vec = x[src] - x[dst]
        coord_update = coord_weight * dir_vec
        x_agg = torch.zeros_like(x)
        x_agg = x_agg.index_add(0, dst, coord_update)
        return h + h_agg, x + x_agg

class EquivariantDecoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, edge_dim, num_layers, cutoff):
        super().__init__()
        self.layers = nn.ModuleList([EGNNLayer(node_dim, hidden_dim, edge_dim) for _ in range(num_layers)])
        self.cutoff = cutoff
        self.init_coord = nn.Linear(node_dim, 3)

    def build_edges(self, x):
        if x.shape[0] > 5000:
            edge_index = radius_graph(x, r=self.cutoff, max_num_neighbors=64, flow='source_to_target')
            edge_dist = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=-1)
            return edge_index, edge_dist
        D = torch.cdist(x, x)
        triu = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
        mask = (D < self.cutoff) & triu
        src, dst = torch.where(mask)
        edge_dist = D[src, dst]
        src_full = torch.cat([src, dst])
        dst_full = torch.cat([dst, src])
        edge_dist_full = torch.cat([edge_dist, edge_dist])
        return torch.stack([src_full, dst_full], dim=0), edge_dist_full

    def forward(self, h, initial_coords=None):
        B, N, _ = h.shape
        if initial_coords is None:
            x = self.init_coord(h)
        else:
            x = initial_coords
        x_out = []
        for b in range(B):
            hi = h[b]; xi = x[b]
            ei, ed = self.build_edges(xi)
            for layer in self.layers:
                hi, xi = layer(hi, xi, ei, ed)
            x_out.append(xi)
        return torch.stack(x_out, dim=0)

# ──────────────── CSOC Controller ────────────────
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
        return torch.clamp(T, base_T*0.5, 3000.0)

class DiffRGRefiner:
    def __init__(self, factor=4):
        self.factor = factor
    def forward(self, coords):
        L = coords.shape[0]; f = self.factor; m = L//f*f
        if m==0: return coords
        x = coords[:m].permute(1,0).unsqueeze(0)
        pooled = F.avg_pool1d(x, kernel_size=f, stride=f)
        up = F.interpolate(pooled, size=L, mode='linear', align_corners=True)
        return up.squeeze(0).permute(1,0)

# ──────────────── Sparse graph builders ────────────────
def sparse_edges(coords, cutoff, k):
    if coords.shape[0]==0:
        return torch.empty((2,0), dtype=torch.long, device=coords.device), torch.empty((0,), device=coords.device)
    edge_index = radius_graph(coords, r=cutoff, max_num_neighbors=k, flow='source_to_target')
    edge_dist = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=-1)
    return edge_index, edge_dist

def cross_sparse_edges(coords1, coords2, cutoff, k):
    if coords1.shape[0]==0 or coords2.shape[0]==0:
        return torch.empty((2,0), dtype=torch.long, device=coords1.device), torch.empty((0,), device=coords1.device)
    row, col = radius(coords1, coords2, r=cutoff, max_num_neighbors=k)
    edge_index = torch.stack([row, col], dim=0)
    edge_dist = torch.norm(coords1[row] - coords2[col], dim=-1)
    return edge_index, edge_dist

# ──────────────── Side‑chain builder (fixed closure) ────────────────
def _map_ref(idx):
    # helper for internal coordinate mapping (moved outside loop)
    if idx==-1: return 0
    if idx==-2: return 1
    if idx==-3: return 2
    if idx==-4: return 3
    return 3+idx

def build_sidechain_atoms(ca, seq, chi_angles):
    device = ca.device
    L = ca.shape[0]
    v = ca[1:] - ca[:-1]; v_norm = F.normalize(v, dim=-1, eps=1e-8)
    N = torch.zeros_like(ca); C = torch.zeros_like(ca)
    N[1:] = ca[1:] - 1.45 * v_norm; N[0] = ca[0] - 1.45 * v_norm[0]
    C[:-1] = ca[:-1] + 1.52 * v_norm; C[-1] = ca[-1] + 1.52 * v_norm[-1]

    all_coords, all_types = [], []
    for i, aa in enumerate(seq):
        if aa in ('G','A','C','T','U') and detect_sequence_type(aa) != 'protein':
            # Skip non-protein residues (DNA/RNA) – they are handled separately
            # For now, place dummy atoms to keep shape consistent (or handle in writer)
            # We'll just skip and not add to all_coords (will need masking later)
            continue
        if aa == 'G':
            res_atoms = torch.stack([N[i], ca[i], C[i]], dim=0)
            all_coords.append(res_atoms); all_types.append(['N','CA','C'])
            continue
        n_i, ca_i, c_i = N[i], ca[i], C[i]
        v1 = n_i - ca_i; v2 = c_i - ca_i
        cb_dir = -(v1+v2); cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
        cb_pos = ca_i + 1.53 * cb_dir
        local_atoms = [n_i, ca_i, c_i, cb_pos]
        local_types = ['N','CA','C','CB']
        topo = RESIDUE_TOPOLOGY.get(aa, [])
        chi_idx = 0
        for (atom_name, atom_type, parent_idx, bond_len, bond_ang_deg, ref_tuple, dihedral_ang0) in topo:
            a_idx = _map_ref(ref_tuple[0])
            b_idx = _map_ref(ref_tuple[1])
            c_idx = _map_ref(ref_tuple[2])
            # Ensure indices are valid
            a_idx = min(a_idx, len(local_atoms)-1)
            b_idx = min(b_idx, len(local_atoms)-1)
            c_idx = min(c_idx, len(local_atoms)-1)
            p_a,p_b,p_c = local_atoms[a_idx], local_atoms[b_idx], local_atoms[c_idx]
            bc = p_c - p_b; bc_norm = F.normalize(bc, dim=-1, eps=1e-8)
            ref_vec = torch.tensor([1.0,0.0,0.0], device=device)
            dot = torch.abs(torch.dot(bc_norm, ref_vec))
            if dot > 0.9: ref_vec = torch.tensor([0.0,1.0,0.0], device=device)
            perp = torch.cross(bc_norm, ref_vec, dim=-1)
            if torch.dot(perp, perp) < 1e-12:
                ref_vec = torch.tensor([0.0,0.0,1.0], device=device)
                perp = torch.cross(bc_norm, ref_vec, dim=-1)
            perp = F.normalize(perp, dim=-1, eps=1e-8)
            chi_val = chi_angles[i, chi_idx] if chi_angles is not None else 0.0
            total_angle = dihedral_ang0 + chi_val
            cos_a, sin_a = torch.cos(total_angle), torch.sin(total_angle)
            cross_bn_perp = torch.cross(bc_norm, perp, dim=-1)
            rotated_perp = perp * cos_a + cross_bn_perp * sin_a
            ang = torch.tensor(bond_ang_deg * math.pi/180.0, device=device)
            bond_dir = torch.cos(ang)*bc_norm + torch.sin(ang)*rotated_perp
            new_pos = p_c + bond_len * bond_dir
            local_atoms.append(new_pos); local_types.append(atom_type)
            chi_idx += 1
        all_coords.append(torch.stack(local_atoms, dim=0)); all_types.append(local_types)
    return all_coords, all_types

def get_full_atom_coords_and_types(ca, seq, chi_angles, protein_mask=None):
    res_coords, res_types = build_sidechain_atoms(ca, seq, chi_angles)
    coords_list, types_list, res_idx_list = [], [], []
    for i,(rc,rt) in enumerate(zip(res_coords, res_types)):
        coords_list.append(rc); types_list.extend(rt)
        res_idx_list.append(torch.full((rc.shape[0],), i, dtype=torch.long, device=ca.device))
    if not coords_list:
        return torch.empty((0,3), device=ca.device), [], torch.empty(0, device=ca.device)
    all_coords = torch.cat(coords_list, dim=0)
    res_indices = torch.cat(res_idx_list, dim=0)
    return all_coords, types_list, res_indices

# ──────────────── Energy functions (v30.1.1.1 chain‑aware) ────────────────
def reconstruct_backbone(ca):
    L = ca.shape[0]; v = ca[1:] - ca[:-1]; v_norm = F.normalize(v, dim=-1, eps=1e-8)
    N = torch.zeros_like(ca); C = torch.zeros_like(ca)
    N[1:] = ca[1:] - 1.45 * v_norm; N[0] = ca[0] - 1.45 * v_norm[0]
    C[:-1] = ca[:-1] + 1.52 * v_norm; C[-1] = ca[-1] + 1.52 * v_norm[-1]
    offset = torch.tensor([0.0,1.24,0.0], device=ca.device)
    O = torch.zeros_like(ca)
    for i in range(L):
        if i < L-1:
            ca_c = C[i]-ca[i]; ca_n = N[i]-ca[i]
            perp = torch.cross(ca_c, ca_n, dim=-1)
            perp_norm = torch.norm(perp)
            if perp_norm > 1e-6: perp = perp/perp_norm
            O[i] = C[i] + 1.24 * perp
        else: O[i] = C[i] + offset
    return {'N':N, 'CA':ca, 'C':C, 'O':O}

def dihedral_angle(p0,p1,p2,p3):
    b0 = p1-p0; b1 = p2-p1; b2 = p3-p2
    b1n = F.normalize(b1, dim=-1, eps=1e-8)
    v = b0 - (b0*b1n).sum(-1,keepdim=True)*b1n
    w = b2 - (b2*b1n).sum(-1,keepdim=True)*b1n
    x = (v*w).sum(-1)
    y = torch.cross(b1n, v, dim=-1)
    y = (y*w).sum(-1)
    return torch.atan2(y+1e-8, x+1e-8)

def compute_phi_psi(atoms, mask=None):
    N, CA, C = atoms['N'], atoms['CA'], atoms['C']
    L = CA.shape[0]
    phi = torch.zeros(L, device=CA.device); psi = torch.zeros(L, device=CA.device)
    if L > 2:
        phi[1:-1] = dihedral_angle(C[:-2], N[1:-1], CA[1:-1], C[1:-1])
        psi[1:-1] = dihedral_angle(N[1:-1], CA[1:-1], C[1:-1], N[2:])
    return phi*180.0/math.pi, psi*180.0/math.pi

def energy_bond(ca, alpha, cfg, mask=None):
    if mask is not None and not mask.any(): return torch.tensor(0.0, device=ca.device)
    target = 3.8 * (1.0 + cfg.alpha_mod_bond * (alpha - 1.0))
    target_pair = 0.5 * (target[1:] + target[:-1])
    d = torch.norm(ca[1:] - ca[:-1], dim=-1)
    if mask is not None:
        # apply mask to bond pairs (mask both residues)
        bond_mask = mask[1:] & mask[:-1]
        if bond_mask.sum()==0: return torch.tensor(0.0, device=ca.device)
        return cfg.w_bond * ((d[bond_mask] - target_pair[bond_mask])**2).mean()
    return cfg.w_bond * ((d - target_pair) ** 2).mean()

def energy_angle(ca, alpha, cfg, mask=None):
    if len(ca) < 3: return torch.tensor(0.0, device=ca.device)
    v1 = ca[:-2] - ca[1:-1]; v2 = ca[2:] - ca[1:-1]
    v1n = F.normalize(v1, dim=-1, eps=1e-8); v2n = F.normalize(v2, dim=-1, eps=1e-8)
    cos_ang = (v1n * v2n).sum(-1)
    target_angle = cfg.angle_target_rad * (1.0 + cfg.alpha_mod_angle * (alpha[1:-1] - 1.0))
    cos_target = torch.cos(target_angle)
    if mask is not None:
        ang_mask = mask[1:-1]
        if ang_mask.sum()==0: return torch.tensor(0.0, device=ca.device)
        return cfg.w_angle * ((cos_ang[ang_mask] - cos_target[ang_mask])**2).mean()
    return cfg.w_angle * ((cos_ang - cos_target) ** 2).mean()

def energy_rama_vectorized(phi, psi, seq, alpha, cfg, mask=None):
    L = len(seq); device = phi.device
    phi0, psi0, width = torch.zeros(L, device=device), torch.zeros(L, device=device), torch.zeros(L, device=device)
    for i, aa in enumerate(seq):
        prior = RAMACHANDRAN_PRIORS.get(aa, RAMACHANDRAN_PRIORS['general'])
        phi0[i], psi0[i], width[i] = prior['phi'], prior['psi'], prior['width']
    width_eff = width * (1.0 + cfg.alpha_mod_rama * (alpha - 1.0))
    dphi = (phi - phi0) / (width_eff + 1e-8)
    dpsi = (psi - psi0) / (width_eff + 1e-8)
    loss = (dphi**2 + dpsi**2)
    if mask is None:
        mask = torch.ones(L, device=device, dtype=torch.bool)
    mask[0], mask[-1] = False, False
    if mask.sum()==0: return torch.tensor(0.0, device=device)
    return cfg.w_rama * (loss * mask.float()).sum() / mask.sum()

def energy_clash_sparse(ca, alpha, edge_index, edge_dist, cfg, mask=None):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=ca.device)
    idx_i, idx_j = edge_index[0], edge_index[1]
    seq_dist = torch.abs(idx_i - idx_j)
    mask_edge = seq_dist > 2
    if not mask_edge.any(): return torch.tensor(0.0, device=ca.device)
    src, dst = idx_i[mask_edge], idx_j[mask_edge]
    if mask is not None:
        keep = mask[src] & mask[dst]
        if not keep.any(): return torch.tensor(0.0, device=ca.device)
        src, dst = src[keep], dst[keep]
    di = edge_dist[mask_edge][keep] if mask is not None else edge_dist[mask_edge]
    radius_i = cfg.clash_radius * (1.0 + cfg.alpha_mod_clash * (alpha[src] - 1.0))
    radius_j = cfg.clash_radius * (1.0 + cfg.alpha_mod_clash * (alpha[dst] - 1.0))
    radius = 0.5 * (radius_i + radius_j)
    clash = torch.relu(radius - di)
    return cfg.w_clash * (clash ** 2).mean() if clash.numel()>0 else torch.tensor(0.0, device=ca.device)

def energy_electro_sparse(ca, seq, edge_index, edge_dist, cfg, mask=None):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=ca.device)
    q = torch.tensor([RESIDUE_CHARGE.get(a, 0.0) for a in seq], device=ca.device)
    qi, qj = q[edge_index[0]], q[edge_index[1]]
    if mask is not None:
        keep = mask[edge_index[0]] & mask[edge_index[1]]
        if not keep.any(): return torch.tensor(0.0, device=ca.device)
        qi, qj = qi[keep], qj[keep]
        r = torch.clamp(edge_dist[keep], min=1e-6)
    else:
        r = torch.clamp(edge_dist, min=1e-6)
    E = qi * qj * torch.exp(-0.1 * r) / (80.0 * r)
    return cfg.w_electro * E.mean()

def energy_solvent_sparse(ca, seq, edge_index, edge_dist, cfg, mask=None):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=ca.device)
    src = edge_index[0]
    counts = torch.zeros(ca.shape[0], device=ca.device)
    counts = counts.index_add(0, src, torch.ones_like(src, dtype=torch.float))
    if mask is not None: counts = counts * mask.float()
    burial = 1.0 - torch.exp(-counts / 20.0)
    hydro = torch.tensor([HYDROPHOBICITY.get(a, 0.0) for a in seq], device=ca.device)
    exposed = torch.where(hydro > 0, hydro * (1.0 - burial), torch.zeros_like(burial))
    buried = torch.where(hydro <= 0, -hydro * burial, torch.zeros_like(burial))
    return cfg.w_solvent * (exposed + buried).mean()

def energy_rotamer_sparse(ca, atoms, seq, edge_index, cfg, mask=None):
    L = ca.shape[0]
    if edge_index.numel() == 0: return torch.tensor(0.0, device=ca.device)
    # only consider residues with mask (protein)
    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=ca.device)
    valid = mask & torch.tensor([(aa != 'G' and i > 0 and i < L-1) for i, aa in enumerate(seq)], device=ca.device)
    if not valid.any(): return torch.tensor(0.0, device=ca.device)
    N, C = atoms['N'], atoms['C']
    ca_m = ca[valid]; n_m = N[valid]; c_m = C[valid]
    v1 = n_m - ca_m; v2 = c_m - ca_m
    cb_dir = -(v1 + v2); cb_dir = F.normalize(cb_dir, dim=-1, eps=1e-8)
    ideal_cb = ca_m + 1.8 * cb_dir
    global_idx = torch.where(valid)[0]
    edge_src, edge_dst = edge_index[0], edge_index[1]
    is_valid_src = valid[edge_src]
    src_idx = torch.where(is_valid_src)[0]
    if src_idx.numel()==0: return torch.tensor(0.0, device=ca.device)
    src_global = edge_src[src_idx]; dst_global = edge_dst[src_idx]
    global_to_valid = torch.full((L,), -1, device=ca.device, dtype=torch.long)
    global_to_valid[valid] = torch.arange(valid.sum(), device=ca.device)
    src_valid = global_to_valid[src_global]
    ca_dst = ca[dst_global]; cb_src = ideal_cb[src_valid]
    dists = torch.norm(cb_src - ca_dst, dim=-1)
    not_self = (src_global != dst_global)
    dists = dists[not_self]; src_valid = src_valid[not_self]
    if dists.numel()==0: return torch.tensor(0.0, device=ca.device)
    min_per_valid = torch.full((valid.sum(),), float('inf'), device=ca.device)
    min_per_valid = torch.scatter_reduce(min_per_valid, 0, src_valid, dists, reduce='amin')
    min_per_valid[min_per_valid == float('inf')] = 10.0
    penalty = torch.relu(4.0 - min_per_valid)
    return cfg.w_rotamer * penalty.mean()

def hbond_energy_from_sparse(O, N, C, alpha, edge_index, edge_dist, cfg, mask=None):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=O.device)
    src = edge_index[0]; dst = edge_index[1]
    if mask is not None:
        keep = mask[src] & mask[dst]
        if not keep.any(): return torch.tensor(0.0, device=O.device)
        src, dst = src[keep], dst[keep]
        edge_dist = edge_dist[keep]
    vec_co = O[src] - C[src]; vec_no = N[dst] - O[src]
    alignment = F.cosine_similarity(vec_co, vec_no, dim=-1, eps=1e-8)
    ideal_dist = 2.9 * (1.0 + cfg.alpha_mod_hbond * (alpha[src].unsqueeze(1) - 1.0)).squeeze()
    E = -alignment * torch.exp(-((edge_dist - ideal_dist) / 0.3) ** 2)
    return E.mean()

def energy_lj_full(all_coords, all_types, res_indices, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=all_coords.device)
    src, dst = edge_index[0], edge_index[1]
    lj = cfg.lj_params
    sigmas = torch.zeros(len(all_types), device=all_coords.device)
    epsilons = torch.zeros(len(all_types), device=all_coords.device)
    for i, t in enumerate(all_types):
        s, e = lj.get(t, (1.9, 0.1))
        sigmas[i] = s; epsilons[i] = e
    sigma_src, sigma_dst = sigmas[src], sigmas[dst]
    eps_src, eps_dst = epsilons[src], epsilons[dst]
    sigma = 0.5 * (sigma_src + sigma_dst)
    eps = torch.sqrt(eps_src * eps_dst)
    r = torch.clamp(edge_dist, min=1e-4)
    inv_r = 1.0 / r; inv_r6 = inv_r ** 6; inv_r12 = inv_r6 ** 2
    lj_energy = 4.0 * eps * ((sigma * inv_r) ** 12 - (sigma * inv_r) ** 6)
    return cfg.w_lj * lj_energy.mean()

def energy_coulomb_full(all_coords, all_types, res_indices, edge_index, edge_dist, cfg):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=all_coords.device)
    src, dst = edge_index[0], edge_index[1]
    q = torch.tensor([cfg.charge_map.get(t, 0.0) for t in all_types], device=all_coords.device)
    qi, qj = q[src], q[dst]
    r = torch.clamp(edge_dist, min=1e-4)
    dielectric = 4.0 * r
    coulomb = 332.0637 * qi * qj / (dielectric * r)
    return cfg.w_coulomb * coulomb.mean()

def energy_coulomb_pme(all_coords, all_types, cfg):
    # unchanged, but could be used for full atom
    device = all_coords.device
    N = all_coords.shape[0]
    if N < 2: return torch.tensor(0.0, device=device)
    q = torch.tensor([cfg.charge_map.get(t, 0.0) for t in all_types], device=device)
    max_extent = (all_coords.max(dim=0).values - all_coords.min(dim=0).values).max()
    box = max_extent * 1.5 + 5.0
    alpha = cfg.pme_alpha; grid = cfg.pme_grid
    frac = (all_coords + box/2) / box; frac = frac % 1.0
    grid_size = [grid, grid, grid]
    rho = torch.zeros(grid_size, device=device)
    indices = (frac * grid).long().clamp(0, grid-1)
    rho[indices[:,0], indices[:,1], indices[:,2]] += q
    rho_k = torch.fft.fftn(rho)
    kx = torch.fft.fftfreq(grid, d=1.0/grid, device=device) * 2*math.pi / box
    ky = torch.fft.fftfreq(grid, d=1.0/grid, device=device) * 2*math.pi / box
    kz = torch.fft.fftfreq(grid, d=1.0/grid, device=device) * 2*math.pi / box
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2; k2[0,0,0] = 1.0
    factor = torch.exp(-k2 / (4*alpha**2)) / k2; factor[0,0,0] = 0.0
    pot_k = rho_k * factor
    pot = torch.fft.ifftn(pot_k).real
    phi = pot[indices[:,0], indices[:,1], indices[:,2]]
    energy = 0.5 * torch.dot(q, phi) * 332.0637
    self_energy = -alpha / math.sqrt(math.pi) * torch.sum(q**2) * 332.0637
    return cfg.w_coulomb * (energy + self_energy)

def energy_torsion_chi(chi_angles, seq, cfg):
    L = len(seq); device = chi_angles.device; max_chi = chi_angles.shape[1]
    energy = 0.0
    for i, aa in enumerate(seq):
        nchi = RESIDUE_NCHI.get(aa, 0)
        for c in range(min(nchi, max_chi)):
            chi = chi_angles[i, c]
            energy = energy + 0.5 * (1.0 - torch.cos(3.0 * chi))
    return cfg.w_torsion * energy / max(1, L)

def sparse_soc_energy(ca, alpha, edge_index, edge_dist, cfg, mask=None):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=ca.device)
    src, dst = edge_index[0], edge_index[1]
    if mask is not None:
        keep = mask[src] & mask[dst]
        if not keep.any(): return torch.tensor(0.0, device=ca.device)
        src, dst = src[keep], dst[keep]; edge_dist = edge_dist[keep]
    ai, aj = alpha[src], alpha[dst]
    a = 0.5 * (ai + aj)
    safe_dist = torch.clamp(edge_dist, min=1e-6)
    K = torch.exp(-a * torch.log(safe_dist)) * torch.exp(-edge_dist / cfg.kernel_lambda)
    E = -K * torch.exp(-edge_dist / 8.0)
    return cfg.w_soc_contact * E.mean()

def compute_avalanche_gradient(ca, alpha, edge_index, edge_dist, cfg):
    if ca.grad is None or edge_index.numel() == 0: return 0.0
    stress = ca.grad.detach()
    stressed = torch.norm(stress, dim=-1) > cfg.avalanche_threshold
    if not stressed.any(): return torch.zeros_like(ca)
    src, dst = edge_index[0], edge_index[1]
    ai, aj = alpha[src], alpha[dst]
    a = 0.5 * (ai + aj)
    safe_dist = torch.clamp(edge_dist, min=1e-6)
    K_edge = torch.exp(-a * torch.log(safe_dist)) * torch.exp(-edge_dist / cfg.kernel_lambda)
    direction = torch.zeros_like(ca)
    stressed_idx = torch.where(stressed)[0]
    grad_stressed = stress[stressed_idx]
    norm = torch.norm(grad_stressed, dim=-1, keepdim=True)
    direction[stressed_idx] = -grad_stressed / (norm + 1e-8)
    src_stressed = stressed[src]
    if not src_stressed.any(): return torch.zeros_like(ca)
    edge_K = K_edge[src_stressed]
    edge_dst = dst[src_stressed]; edge_src = src[src_stressed]
    direction_src = direction[edge_src]
    grad_contrib = torch.zeros_like(ca)
    grad_contrib.index_add_(0, edge_dst, -cfg.w_avalanche * edge_K.unsqueeze(-1) * direction_src)
    return grad_contrib

def alpha_regularisation(alpha, cfg):
    entropy = -(alpha * torch.log(alpha + 1e-8)).mean()
    diff = alpha[1:] - alpha[:-1]
    smooth = (diff ** 2).mean()
    return cfg.w_alpha_entropy * entropy + cfg.w_alpha_smooth * smooth

def chain_break_energy(ca, chain_boundaries, cfg, mask=None):
    if not chain_boundaries: return torch.tensor(0.0, device=ca.device)
    energy = 0.0
    for start in chain_boundaries:
        dist = torch.norm(ca[start] - ca[start-1], dim=-1)
        energy = energy + torch.relu(dist - 5.0)
    return cfg.chain_break_weight * energy

# ──────────────── Total energy (v30.1.1.1 chain‑aware) ────────────────
def _total_physics_energy_v30_1_1_impl(ca, seq, alpha, chi_angles,
                                       edge_index_ca, edge_dist_ca,
                                       edge_index_hbond, edge_dist_hbond,
                                       chain_boundaries, cfg,
                                       chain_types=None):
    # Detect chain types if not provided
    if chain_types is None:
        chain_types = [detect_sequence_type(seq[i:i+1]) for i in range(len(seq))]  # per-residue rough
    protein_mask = torch.tensor([t=='protein' for t in chain_types], device=ca.device, dtype=torch.bool)
    dna_mask = torch.tensor([t in ('dna','rna') for t in chain_types], device=ca.device, dtype=torch.bool)

    e = 0.0
    # Protein energy terms (only for protein residues)
    if protein_mask.any():
        # subset ca for protein if needed, but many functions work with mask
        atoms = reconstruct_backbone(ca)
        phi, psi = compute_phi_psi(atoms)
        e += energy_bond(ca, alpha, cfg, mask=protein_mask)
        e += energy_angle(ca, alpha, cfg, mask=protein_mask)
        e += energy_rama_vectorized(phi, psi, seq, alpha, cfg, mask=protein_mask)
        e += energy_clash_sparse(ca, alpha, edge_index_ca, edge_dist_ca, cfg, mask=protein_mask)
        e += cfg.w_hbond * hbond_energy_from_sparse(atoms['O'], atoms['N'], atoms['C'], alpha,
                                                    edge_index_hbond, edge_dist_hbond, cfg, mask=protein_mask)
        e += energy_electro_sparse(ca, seq, edge_index_ca, edge_dist_ca, cfg, mask=protein_mask)
        e += energy_solvent_sparse(ca, seq, edge_index_ca, edge_dist_ca, cfg, mask=protein_mask)
        e += energy_rotamer_sparse(ca, atoms, seq, edge_index_ca, cfg, mask=protein_mask)
        e += alpha_regularisation(alpha, cfg)
        e += chain_break_energy(ca, chain_boundaries, cfg, mask=protein_mask)

        if chi_angles is not None:
            # Build full atom only for protein residues
            prot_idx = torch.where(protein_mask)[0]
            if len(prot_idx) > 0:
                prot_ca = ca[prot_idx]
                prot_seq = "".join([seq[i] for i in prot_idx.tolist()])
                prot_chi = chi_angles[prot_idx] if chi_angles is not None else None
                all_coords, all_types, _ = get_full_atom_coords_and_types(prot_ca, prot_seq, prot_chi)
                if all_coords.shape[0] > 0:
                    edge_idx_full, edge_dist_full = sparse_edges(all_coords, cfg.sparse_cutoff, cfg.max_neighbors)
                    e += energy_lj_full(all_coords, all_types, _, edge_idx_full, edge_dist_full, cfg)
                    e += energy_coulomb_full(all_coords, all_types, _, edge_idx_full, edge_dist_full, cfg)
                    if cfg.use_pme:
                        e += energy_coulomb_pme(all_coords, all_types, cfg)
                    e += energy_torsion_chi(prot_chi, prot_seq, cfg)

        if edge_index_ca is not None and edge_index_ca.numel() > 0:
            e += sparse_soc_energy(ca, alpha, edge_index_ca, edge_dist_ca, cfg, mask=protein_mask)

    # DNA/RNA energy terms
    if dna_mask.any():
        dna_idx = torch.where(dna_mask)[0]
        if len(dna_idx) > 0:
            dna_ca = ca[dna_idx]  # these are actually C4' coordinates
            dna_seq = "".join([seq[i] for i in dna_idx.tolist()])
            dna_eng = cfg.dna_rna_eng
            e += dna_eng(dna_ca, dna_seq, edge_index_ca, edge_dist_ca)  # pass edges for LJ/Coulomb

    return e

# torch.compile guard
_total_physics_energy_v30_1_1 = _total_physics_energy_v30_1_1_impl
if hasattr(torch, 'compile') and V30_1_1Config().use_torch_compile:
    try:
        _total_physics_energy_v30_1_1 = torch.compile(_total_physics_energy_v30_1_1_impl, mode='max-autotune')
    except:
        pass

def total_physics_energy_v30_1_1(*args, **kwargs):
    return _total_physics_energy_v30_1_1(*args, **kwargs)

# ──────────────── Core Model (unchanged mostly) ────────────────
class CSOCSSC_V30_1_1(nn.Module):
    def __init__(self, cfg: V30_1_1Config):
        super().__init__()
        self.cfg = cfg
        if cfg.use_msa:
            self.msa_encoder = MSAEncoder(cfg.msa_dim, n_layers=4, heads=4, out_dim=cfg.dim)
            self.seq_encoder = None
        else:
            self.msa_encoder = None
            self.seq_encoder = FlashSequenceEncoder(cfg.dim, cfg.depth, cfg.heads, cfg.ff_mult)
        self.decoder = EquivariantDecoder(cfg.dim, cfg.egnn_hidden, cfg.egnn_edge_dim,
                                          cfg.egnn_layers, cfg.egnn_cutoff)
        self.alpha_field = AdaptiveAlphaField(cfg.dim)
        self.csoc = CSOCController()
        self.rg = DiffRGRefiner(cfg.rg_factor) if cfg.use_rg else None

    def forward(self, seq_ids, initial_coords=None, msa=None):
        if self.cfg.use_msa and msa is not None:
            latent = self.msa_encoder(msa)
        else:
            latent = self.seq_encoder(seq_ids)
        coords = self.decoder(latent, initial_coords)
        alpha = self.alpha_field(latent)
        return coords, alpha

    def predict_multimer(self, sequences, initial_coords_list=None, msa=None, chain_types=None):
        self.eval()
        all_seq = "".join(sequences)
        ids = torch.tensor([AA_TO_ID.get(a,20) for a in all_seq], dtype=torch.long, device=self.cfg.device).unsqueeze(0)
        with torch.no_grad():
            if self.cfg.use_msa and msa is not None:
                latent = self.msa_encoder(msa)
            else:
                latent = self.seq_encoder(ids)
            init = None
            if initial_coords_list:
                init = torch.cat([torch.tensor(c, device=self.cfg.device) for c in initial_coords_list], dim=0).unsqueeze(0)
            coords, alpha = self.decoder(latent, init)
        coords_np = coords.squeeze(0).cpu().numpy(); alpha_np = alpha.squeeze(0).cpu().numpy()
        chains_ca = []
        idx = 0
        for s in sequences:
            l = len(s); chains_ca.append(coords_np[idx:idx+l]); idx += l
        return chains_ca, alpha_np

    def refine_multimer(self, sequences, init_coords_list=None, steps=None, logger=None, msa=None, chain_types=None):
        if steps is None: steps = self.cfg.refine_steps
        self.eval()
        device = torch.device(self.cfg.device)
        all_seq = "".join(sequences)
        L = len(all_seq)
        boundaries = []
        idx = 0
        for s in sequences[:-1]:
            idx += len(s); boundaries.append(idx)

        # detect chain types if not provided
        if chain_types is None:
            chain_types = []
            for seq in sequences:
                ct = detect_sequence_type(seq)
                chain_types.extend([ct]*len(seq))

        if init_coords_list is not None:
            ca_np = np.concatenate([c - c.mean(axis=0) for c in init_coords_list], axis=0)
            if self.cfg.zero_copy_pinned and device.type == 'cuda':
                ca = torch.from_numpy(ca_np).pin_memory().to(device, non_blocking=True).float().requires_grad_(True)
            else:
                ca = torch.tensor(ca_np, device=device).float().requires_grad_(True)
            with torch.no_grad():
                if self.cfg.use_msa and msa is not None:
                    latent = self.msa_encoder(msa)
                else:
                    ids = torch.tensor([AA_TO_ID.get(a,20) for a in all_seq], dtype=torch.long, device=device).unsqueeze(0)
                    latent = self.seq_encoder(ids)
                alpha = self.alpha_field(latent).squeeze(0)
        else:
            with torch.no_grad():
                chains_ca, alpha_np = self.predict_multimer(sequences, msa=msa, chain_types=chain_types)
            ca_np = np.concatenate(chains_ca, axis=0)
            ca = torch.tensor(ca_np, device=device, requires_grad=True)
            alpha = torch.tensor(alpha_np, device=device)

        max_chi = MAX_CHI
        chi = nn.Parameter(torch.zeros((L, max_chi), device=device).uniform_(-math.pi, math.pi))

        edge_index_ca, edge_dist_ca = sparse_edges(ca, self.cfg.sparse_cutoff, self.cfg.max_neighbors)
        atoms = reconstruct_backbone(ca)
        edge_index_hbond, edge_dist_hbond = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.cfg.max_neighbors)

        opt = torch.optim.Adam([ca, chi], lr=self.cfg.lr)
        scaler = GradScaler(enabled=self.cfg.use_amp)
        energy_history = []

        for step in range(steps):
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=self.cfg.use_amp):
                e_phys = total_physics_energy_v30_1_1(ca, all_seq, alpha, chi,
                                                      edge_index_ca, edge_dist_ca,
                                                      edge_index_hbond, edge_dist_hbond,
                                                      boundaries, self.cfg,
                                                      chain_types=chain_types)
                loss = e_phys

            scaler.scale(loss).backward()
            if self.cfg.w_avalanche > 0 and edge_index_ca.numel() > 0:
                av_grad = compute_avalanche_gradient(ca, alpha, edge_index_ca, edge_dist_ca, self.cfg)
                if torch.is_tensor(av_grad): ca.grad = ca.grad + av_grad

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([ca, chi], max_norm=10.0)
            scaler.step(opt); scaler.update()

            sigma = self.csoc.sigma(ca.detach())
            T = self.csoc.temperature(sigma, self.cfg.temp_base, self.cfg.sigma_target)
            noise_scale = math.sqrt(2 * self.cfg.friction * T.item() / 300.0) * self.cfg.lr
            with torch.no_grad():
                ca.add_(torch.randn_like(ca) * noise_scale)
                chi.data.add_(torch.randn_like(chi) * noise_scale * 0.5)

            if step>0 and step%self.cfg.rebuild_interval==0:
                del edge_index_ca, edge_dist_ca, edge_index_hbond, edge_dist_hbond
                torch.cuda.empty_cache()
                edge_index_ca, edge_dist_ca = sparse_edges(ca.detach(), self.cfg.sparse_cutoff, self.cfg.max_neighbors)
                atoms = reconstruct_backbone(ca.detach())
                edge_index_hbond, edge_dist_hbond = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.cfg.max_neighbors)

            if self.rg and step>0 and step%self.cfg.rg_interval==0:
                ca.data = self.rg.forward(ca.data)

            if step%50==0 and logger:
                logger.info(f"refine {step:04d}  loss={loss.item():.4f}  phys={e_phys.item():.4f} "
                            f"σ={sigma.item():.3f}  T={T.item():.1f}")
                energy_history.append(loss.item())

        return ca.detach().cpu().numpy(), chi.detach().cpu().numpy(), energy_history

# ──────────────── Multi‑GPU wrapper ────────────────
def run_replica(local_rank, cfg, sequences, init_coords_list, msa, out_pdb, return_dict):
    device = f"cuda:{local_rank}"
    cfg.device = device
    model = CSOCSSC_V30_1_1(cfg).to(device)
    ckpt_path = os.path.join(cfg.checkpoint_dir, "v30_1_1_pretrained.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    refined_ca, refined_chi, _ = model.refine_multimer(sequences, init_coords_list, steps=cfg.refine_steps, msa=msa)
    return_dict[local_rank] = (refined_ca, refined_chi)

def refine_multimer_multi_gpu(cfg, sequences, init_coords_list, msa, out_pdb):
    num_gpus = min(cfg.num_replicas, torch.cuda.device_count())
    if num_gpus <= 1:
        model = CSOCSSC_V30_1_1(cfg).to(torch.device(cfg.device))
        refined_ca, refined_chi, _ = model.refine_multimer(sequences, init_coords_list, steps=cfg.refine_steps, msa=msa)
        return refined_ca, refined_chi

    import torch.multiprocessing as mp
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=run_replica, args=(rank, cfg, sequences, init_coords_list, msa, out_pdb, return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    best_energy = float('inf')
    best_ca, best_chi = None, None
    for rank, (ca, chi) in return_dict.items():
        ca_t = torch.tensor(ca, dtype=torch.float32)
        chi_t = torch.tensor(chi, dtype=torch.float32)
        e_phys = total_physics_energy_v30_1_1(ca_t, "".join(sequences), torch.ones(len(ca_t)), chi_t,
                                             *sparse_edges(ca_t, cfg.sparse_cutoff, cfg.max_neighbors),
                                             *cross_sparse_edges(reconstruct_backbone(ca_t)['O'], reconstruct_backbone(ca_t)['N'], 3.5, cfg.max_neighbors),
                                             [], cfg)
        if e_phys.item() < best_energy:
            best_energy = e_phys.item()
            best_ca, best_chi = ca, chi
    return best_ca, best_chi

# ──────────────── Training (fixed) ────────────────
def train_model(model, dataloader, cfg, logger, use_msa=False):
    device = torch.device(cfg.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.use_amp)
    accumulation_steps = cfg.gradient_accumulation_steps
    for epoch in range(cfg.epochs):
        if cfg.is_distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        total_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            if use_msa:
                seq_ids, target_coords, msa = batch
                msa = msa.to(device, non_blocking=True)
            else:
                seq_ids, target_coords = batch
                msa = None
            seq_ids = seq_ids.to(device, non_blocking=True)
            target_coords = target_coords.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=cfg.use_amp):
                pred_coords, pred_alpha = model(seq_ids, msa=msa)
                coord_loss = F.mse_loss(pred_coords, target_coords)
                alpha_reg = 0.001 * ((pred_alpha[:,1:] - pred_alpha[:,:-1])**2).mean()
                loss = (coord_loss + alpha_reg) / accumulation_steps

            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps

        # if any remaining gradients (for incomplete accumulation)
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if cfg.local_rank in [-1, 0]:
            logger.info(f"Epoch {epoch+1:03d}/{cfg.epochs}  MSE={total_loss/len(dataloader):.4f}")

    if cfg.local_rank in [-1, 0]:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(cfg.checkpoint_dir, "v30_1_1_pretrained.pt")
        state_dict = model.module.state_dict() if cfg.is_distributed else model.state_dict()
        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")

# ──────────────── PDB writer (handles DNA/RNA) ────────────────
def write_full_pdb_multimer(ca, seq, chi_angles, filename, chain_ids=None, chain_types=None):
    device = 'cpu'
    ca_t = torch.tensor(ca, dtype=torch.float32)
    chi_t = torch.tensor(chi_angles, dtype=torch.float32)
    # detect chain types if not provided
    if chain_types is None:
        chain_types = [detect_sequence_type(seq[i:i+1]) for i in range(len(seq))]
    with open(filename, 'w') as f:
        atom_serial = 1
        for i, aa in enumerate(seq):
            ct = chain_types[i]
            chain = chain_ids[i] if chain_ids else 'A'
            if ct == 'protein':
                # Write standard protein atoms
                atoms = reconstruct_backbone(ca_t)
                N, C, O = atoms['N'], atoms['C'], atoms['O']
                f.write(f"ATOM  {atom_serial:5d}  N   {aa:3s} {chain}{i+1:4d}    "
                        f"{N[i,0]:8.3f}{N[i,1]:8.3f}{N[i,2]:8.3f}  1.00  0.00           N\n")
                atom_serial += 1
                f.write(f"ATOM  {atom_serial:5d}  CA  {aa:3s} {chain}{i+1:4d}    "
                        f"{ca[i,0]:8.3f}{ca[i,1]:8.3f}{ca[i,2]:8.3f}  1.00  0.00           C\n")
                atom_serial += 1
                f.write(f"ATOM  {atom_serial:5d}  C   {aa:3s} {chain}{i+1:4d}    "
                        f"{C[i,0]:8.3f}{C[i,1]:8.3f}{C[i,2]:8.3f}  1.00  0.00           C\n")
                atom_serial += 1
                f.write(f"ATOM  {atom_serial:5d}  O   {aa:3s} {chain}{i+1:4d}    "
                        f"{O[i,0]:8.3f}{O[i,1]:8.3f}{O[i,2]:8.3f}  1.00  0.00           O\n")
                atom_serial += 1
                # sidechain (if any)
                if aa != 'G':
                    # build sidechain for this residue only
                    all_coords, all_types, res_indices = get_full_atom_coords_and_types(
                        ca_t[i:i+1], [aa], chi_t[i:i+1] if chi_t is not None else None)
                    # write sidechain atoms (skip first 4 backbone atoms)
                    for k in range(4, all_coords.shape[0]):
                        at_name = all_types[k]; x,y,z = all_coords[k]
                        f.write(f"ATOM  {atom_serial:5d}  {at_name:4s}{aa:3s} {chain}{i+1:4d}    "
                                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {at_name[0]}\n")
                        atom_serial += 1
            else:
                # DNA/RNA: write as C4' only for now (can be extended)
                at_name = "C4'"
                x,y,z = ca[i,0], ca[i,1], ca[i,2]
                f.write(f"HETATM{atom_serial:5d}  {at_name:4s} {aa:3s} {chain}{i+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                atom_serial += 1
        f.write("END\n")
    print(f"Full‑atom PDB written to {filename}")

# ──────────────── CLI ────────────────
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="CSOC‑SSC V30.1.1.1 – MSA‑Enabled Hybrid Folding Engine")
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train')
    train_parser.add_argument('--pdb_dir', type=str, required=True)
    train_parser.add_argument('--msa_dir', type=str, default=None)
    train_parser.add_argument('--epochs', type=int, default=80)
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument('--accumulation_steps', type=int, default=1)

    refine_parser = sub.add_parser('refine')
    refine_parser.add_argument('--seq', nargs='+', type=str)
    refine_parser.add_argument('--pdb', type=str, default=None)
    refine_parser.add_argument('--init', nargs='+', type=str)
    refine_parser.add_argument('--msa', type=str, default=None)
    refine_parser.add_argument('--out', type=str, default='refined_v30_1_1.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--checkpoint', type=str, default='v30_1_1_pretrained.pt')
    refine_parser.add_argument('--num_replicas', type=int, default=1)
    refine_parser.add_argument('--lj_params', type=str, default=None)
    refine_parser.add_argument('--charge_params', type=str, default=None)

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    use_msa = False
    if args.command == 'train' and args.msa_dir:
        use_msa = True
    elif args.command == 'refine' and args.msa:
        use_msa = True

    cfg = V30_1_1Config(
        local_rank=local_rank, is_distributed=is_distributed, device=device_str,
        use_msa=use_msa,
        epochs=getattr(args,'epochs',80), batch_size=getattr(args,'batch_size',8),
        gradient_accumulation_steps=getattr(args,'accumulation_steps',1),
        refine_steps=getattr(args,'steps',600), num_replicas=getattr(args,'num_replicas',1),
        lj_param_file=getattr(args,'lj_params',None),
        charge_param_file=getattr(args,'charge_params',None),
        pdb_dir=getattr(args,'pdb_dir',None),
        msa_dir=getattr(args,'msa_dir',None)
    )

    torch.manual_seed(cfg.seed + (local_rank if local_rank>0 else 0))
    np.random.seed(cfg.seed + (local_rank if local_rank>0 else 0))
    random.seed(cfg.seed + (local_rank if local_rank>0 else 0))
    logger = setup_logger("CSOC-SSC_V30.1.1.1", local_rank)

    if local_rank in [-1,0]:
        logger.info("="*60)
        logger.info("CSOC-SSC V30.1.1.1 – MSA‑Enabled Hybrid Folding Engine")
        logger.info(f"PME: {cfg.use_pme}  torch.compile: {cfg.use_torch_compile}  Zero‑Copy: {cfg.zero_copy_pinned}")
        logger.info(f"MSA: {use_msa}  Distributed: {is_distributed}  Device: {device_str}")
        logger.info("="*60)

    if args.command == 'train':
        if local_rank in [-1,0]:
            logger.info("Loading training data...")
        if use_msa:
            dataset = MSAProteinDataset(args.pdb_dir, args.msa_dir)
        else:
            dataset = RealProteinDataset(args.pdb_dir)
        if is_distributed:
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, pin_memory=cfg.zero_copy_pinned)
        else:
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=cfg.zero_copy_pinned)
        model = CSOCSSC_V30_1_1(cfg).to(torch.device(device_str))
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        train_model(model, dataloader, cfg, logger, use_msa=use_msa)

    elif args.command == 'refine':
        if args.pdb:
            logger.info(f"Fetching multimer PDB {args.pdb}...")
            backbones, chain_ids = MultimerPDBFetcher.fetch(args.pdb)
            sequences = [b.seq for b in backbones]
            init_coords_list = [b.ca for b in backbones]
            chain_labels = [b.chain_id for b in backbones]
        elif args.seq:
            sequences = args.seq
            init_coords_list = []
            if args.init:
                for init_file in args.init:
                    coords_list = []
                    with open(init_file) as f:
                        for line in f:
                            if line.startswith('ATOM') and line[12:16].strip()=='CA':
                                coords_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    init_coords_list.append(np.array(coords_list, dtype=np.float32))
            else:
                init_coords_list = None
            chain_labels = None
        else:
            raise ValueError("Must provide --seq or --pdb.")

        msa_tensor = None
        if args.msa:
            msa_np = np.load(args.msa)
            msa_tensor = torch.tensor(msa_np, dtype=torch.float32).unsqueeze(0).to(device_str)

        start_time = time.time()
        if cfg.num_replicas > 1 and torch.cuda.device_count() > 1:
            refined_ca, refined_chi = refine_multimer_multi_gpu(cfg, sequences, init_coords_list, msa_tensor, args.out)
        else:
            model = CSOCSSC_V30_1_1(cfg).to(torch.device(device_str))
            ckpt = args.checkpoint
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=device_str))
                logger.info(f"Loaded weights from {ckpt}")
            else:
                logger.warning("Checkpoint not found; using random weights.")
            refined_ca, refined_chi, _ = model.refine_multimer(sequences, init_coords_list,
                                                               steps=cfg.refine_steps, logger=logger, msa=msa_tensor)
        all_seq = "".join(sequences)
        if chain_labels:
            res_chain = []
            for seq, lbl in zip(sequences, chain_labels):
                res_chain.extend([lbl]*len(seq))
        else:
            res_chain = ['A'] * len(all_seq)
        # detect chain types for writing
        chain_types = []
        for s in sequences:
            ct = detect_sequence_type(s)
            chain_types.extend([ct]*len(s))
        write_full_pdb_multimer(refined_ca, all_seq, refined_chi, args.out, res_chain, chain_types)
        logger.info(f"Full‑atom refined structure saved to {args.out}")
        logger.info(f"Compute Time: {time.time()-start_time:.2f} seconds")

    if is_distributed:
        dist.destroy_process_group()

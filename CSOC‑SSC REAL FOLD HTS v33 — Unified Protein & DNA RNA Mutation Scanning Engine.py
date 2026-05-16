# =============================================================================
# CSOC‑SSC HTS FOLD v33 — Unified Protein & DNA/RNA Mutation Scanning Engine
# =============================================================================
import os, sys, argparse, logging, json, itertools, copy, time, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("HTS-FOLD-v33")

# ──────────────────────────────────────────────────────────────────────────────
# Import CSOC‑SSC v30.1.1.1.2 (Hybrid Engine) – single source for all energies
# ──────────────────────────────────────────────────────────────────────────────
try:
    from csoc_v30_1_1_1_2 import (
        CSOCSSC_V30_1_1, V30_1_1Config, total_physics_energy,
        reconstruct_backbone, sparse_edges, cross_sparse_edges,
        get_full_atom_coords_and_types, build_sidechain_atoms,
        detect_sequence_type, DEFAULT_CHARGE_MAP, DEFAULT_LJ_PARAMS,
        MAX_CHI, RESIDUE_NCHI, AA_VOCAB, AA_TO_ID, AA_3_TO_1,
        FullDNA_RNA_Energy, build_full_dna_rna, build_dna_helix,
        NUCLEOTIDE_LJ, NUCLEOTIDE_CHARGES,
        WC_PAIRS, BASE_STACKING,
    )
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    logger.warning("v30.1.1.1.2 not found. Falling back to separate imports (limited).")
    # Try to import older modules as fallback
    try:
        from csoc_v30_1 import (
            CSOCSSC_V30_1 as CSOCSSC_V30_1_1, V30_1Config as V30_1_1Config,
            total_physics_energy_v30_1 as total_physics_energy,
            reconstruct_backbone, sparse_edges, cross_sparse_edges,
            get_full_atom_coords_and_types, detect_sequence_type,
            MAX_CHI, RESIDUE_NCHI, AA_VOCAB, AA_TO_ID, AA_3_TO_1,
        )
    except:
        raise ImportError("No CSOC‑SSC engine found. Install csoc_v30_1_1_1_2.py first.")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
class HTSConfigV33:
    def __init__(self,
                 pdb_structure: Optional[str] = None,
                 sequence: Optional[str] = None,
                 chain_id: Optional[str] = None,
                 ds_type: str = 'B_DNA',          # only used when building DNA/RNA de novo
                 output_dir: str = "./hts_output_v33",
                 ddg_threshold: float = 0.5,
                 mutation_list: Optional[List[Tuple[int, str]]] = None,
                 scan_full: bool = False,
                 scan_epistasis: bool = False,
                 epistasis_pairs: Optional[List[Tuple[int, int]]] = None,
                 relaxation_steps: int = 30,
                 use_gpu: bool = False,
                 num_gpus: int = 1,
                 lj_param_file: Optional[str] = None,
                 charge_param_file: Optional[str] = None,
                 ):
        self.pdb_structure = pdb_structure
        self.sequence = sequence          # if provided, auto‑detect type
        self.chain_id = chain_id or 'A'
        self.ds_type = ds_type
        self.output_dir = output_dir
        self.ddg_threshold = ddg_threshold
        self.mutation_list = mutation_list
        self.scan_full = scan_full
        self.scan_epistasis = scan_epistasis
        self.epistasis_pairs = epistasis_pairs
        self.relaxation_steps = relaxation_steps
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        self.lj_param_file = lj_param_file
        self.charge_param_file = charge_param_file

# ──────────────────────────────────────────────────────────────────────────────
# Mutation Utilities (auto‑detect alphabet)
# ──────────────────────────────────────────────────────────────────────────────
def get_alphabet(seq: str) -> List[str]:
    """Return list of possible monomers for the given sequence type."""
    seq_type = detect_sequence_type(seq)
    if seq_type == 'protein':
        return list(AA_VOCAB.replace('X', ''))  # 20 amino acids
    elif seq_type == 'dna':
        return ['A', 'C', 'G', 'T']
    elif seq_type == 'rna':
        return ['A', 'C', 'G', 'U']
    else:
        # Fallback
        return list(AA_VOCAB.replace('X', ''))

def is_transition(old: str, new: str, seq_type: str) -> bool:
    """Check if mutation is a transition (purine↔purine, pyrimidine↔pyrimidine for DNA/RNA)."""
    if seq_type in ('dna', 'rna'):
        transitions = {('A','G'), ('G','A'), ('C','T'), ('T','C'),
                       ('A','U'), ('U','A'), ('C','U'), ('U','C')}
        return (old, new) in transitions
    # Protein: treat as always false (no simple classification)
    return False

def is_transversion(old: str, new: str, seq_type: str) -> bool:
    if seq_type in ('dna', 'rna'):
        return not is_transition(old, new, seq_type) and old != new
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Unified HTS Analyzer
# ──────────────────────────────────────────────────────────────────────────────
class HTSAnalyzerV33:
    def __init__(self, config: HTSConfigV33):
        self.config = config
        self.device = torch.device("cuda" if (torch.cuda.is_available() and config.use_gpu) else "cpu")
        # State
        self.sequences = []          # list of per‑chain sequences
        self.chain_types = []        # 'protein', 'dna', 'rna' per chain
        self.full_seq = ""           # concatenated
        self.ca_coords = None        # [total_L, 3] tensor (CA for protein, C4' for DNA/RNA)
        self.chi = None              # sidechain chi angles (protein only)
        self.wt_energy = None
        self.alpha = None
        self.v30_cfg = None
        self.model = None
        # For DNA/RNA full atom building
        self.dna_rna_engines = {}    # chain_idx -> FullDNA_RNA_Energy

    def load_structure(self):
        """Load structure from PDB or build from sequence."""
        if self.config.pdb_structure and os.path.exists(self.config.pdb_structure):
            self._load_from_pdb()
        elif self.config.sequence:
            self._build_from_sequence()
        else:
            raise ValueError("Either --pdb or --seq must be provided.")

        # Set up V30 config and compute WT energy
        self.v30_cfg = V30_1_1Config(
            device=str(self.device),
            use_pme=True,
            use_torch_compile=False,  # safe for HTS
            zero_copy_pinned=True,
            lj_param_file=self.config.lj_param_file,
            charge_param_file=self.config.charge_param_file
        )
        self._compute_wt_energy()

    def _load_from_pdb(self):
        # Use the existing fetch_from_file to get all chains (protein + DNA)
        from csoc_v30_1_1_1_2 import MultimerPDBFetcher
        backbones, chain_ids = MultimerPDBFetcher.fetch_from_file(self.config.pdb_structure)
        if not backbones:
            raise ValueError("No chains found in PDB.")
        # Filter by chain if requested
        if self.config.chain_id:
            backbones = [b for b in backbones if b.chain_id == self.config.chain_id]
            if not backbones:
                raise ValueError(f"Chain {self.config.chain_id} not found.")
        self.sequences = [b.seq for b in backbones]
        self.chain_types = [detect_sequence_type(b.seq) for b in backbones]
        self.full_seq = "".join(b.seq for b in backbones)
        # Concatenate coordinates (CA for protein, CA for DNA? In PDB, DNA chains have CA atoms? No, they have C4'. 
        # The MultimerPDBFetcher currently only reads CA atoms. For DNA chains, we should read C4'.
        # We'll need a specialized reader that extracts CA for protein and C4' for DNA from the same PDB.
        # As a fallback, we'll assume protein-only for now, but for real mixed systems we need a better parser.
        # For v33, we'll implement a simple multi-type PDB reader.
        all_coords = []
        for i, bb in enumerate(backbones):
            if self.chain_types[i] == 'protein':
                # CA coordinates
                coords = torch.tensor(bb.ca, dtype=torch.float32)
            else:
                # DNA/RNA: we need C4' atoms. If the PDB was loaded by fetch_from_file, it also grabbed CA? 
                # We'll need to reload that chain specifically with load_nucleotide_pdb.
                # For now, we raise error for mixed chains – but we can implement.
                logger.warning(f"Chain {bb.chain_id} is DNA/RNA; using C4' reader from csoc_dna_rna module.")
                from csoc_dna_rna_v30_2_1 import load_nucleotide_pdb
                c4, _ = load_nucleotide_pdb(self.config.pdb_structure, chain=bb.chain_id)
                coords = c4
            all_coords.append(coords)
        self.ca_coords = torch.cat([c.to(self.device) for c in all_coords], dim=0)
        # Build chi for protein residues (later we will handle sidechain)
        # For now, we'll set chi to zero for all and rely on backbone energy for speed.
        L = len(self.full_seq)
        self.chi = torch.zeros((L, MAX_CHI), device=self.device)
        # For DNA chains, chi is irrelevant.

    def _build_from_sequence(self):
        seq = self.config.sequence.upper()
        ct = detect_sequence_type(seq)
        self.sequences = [seq]
        self.chain_types = [ct]
        self.full_seq = seq
        if ct == 'protein':
            # Placeholder – use random coords or run model prediction (but for HTS we need a structure)
            raise NotImplementedError("De novo protein structure not supported; provide a PDB.")
        else:
            # Build ideal helix
            from csoc_v30_1_1_1_2 import build_dna_helix
            is_rna = (ct == 'rna')
            if is_rna:
                helix = build_dna_helix(seq, rise=2.8, twist=32.7, radius=9.0)
            else:
                helix = build_dna_helix(seq, rise=3.38, twist=36.0, radius=8.0)
            self.ca_coords = helix.to(self.device)
            self.chi = torch.zeros((len(seq), MAX_CHI), device=self.device)

    def _compute_wt_energy(self):
        self.alpha = torch.ones(len(self.full_seq), device=self.device)
        ei, ed = sparse_edges(self.ca_coords, self.v30_cfg.sparse_cutoff, self.v30_cfg.max_neighbors)
        atoms = reconstruct_backbone(self.ca_coords)
        ei_hb, ed_hb = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.v30_cfg.max_neighbors)
        # Build chain_types list per residue
        chain_types_per_res = []
        for seq, ct in zip(self.sequences, self.chain_types):
            chain_types_per_res.extend([ct] * len(seq))
        self.wt_energy = total_physics_energy(
            self.ca_coords, self.full_seq, self.alpha, self.chi,
            ei, ed, ei_hb, ed_hb,
            [len(s) for s in self.sequences[:-1]] if len(self.sequences)>1 else [],
            self.v30_cfg,
            chain_types=chain_types_per_res
        ).item()
        logger.info(f"WT energy: {self.wt_energy:.4f}")

    def compute_ddg_single(self, chain_idx: int, pos_in_chain: int, new_monomer: str,
                           relax: bool = True) -> Dict:
        """Compute ΔΔG for a single mutation in a specific chain."""
        # Determine global position in full_seq
        offset = sum(len(s) for s in self.sequences[:chain_idx])
        glob_pos = offset + pos_in_chain
        wt_monomer = self.full_seq[glob_pos]
        if wt_monomer == new_monomer:
            return {'position': glob_pos, 'chain': chain_idx, 'wt': wt_monomer,
                    'mut': new_monomer, 'ddg': 0.0, 'type': 'self'}
        mut_full = self.full_seq[:glob_pos] + new_monomer + self.full_seq[glob_pos+1:]
        # Start from WT coords
        mut_coords = self.ca_coords.clone()
        if relax and self.config.relaxation_steps > 0:
            mut_coords, e_mut = self._relax_mutant(mut_coords, mut_full, glob_pos)
        else:
            e_mut = self._compute_energy(mut_coords, mut_full)
        ddg = e_mut - self.wt_energy
        seq_type = self.chain_types[chain_idx]
        mut_type = 'transition' if is_transition(wt_monomer, new_monomer, seq_type) else (
            'transversion' if is_transversion(wt_monomer, new_monomer, seq_type) else 'mutation')
        return {
            'chain': chain_idx,
            'position': pos_in_chain,
            'global_position': glob_pos,
            'wt': wt_monomer,
            'mut': new_monomer,
            'ddg': ddg,
            'relaxed': relax and self.config.relaxation_steps > 0,
            'type': mut_type,
            'e_wt': self.wt_energy,
            'e_mut': e_mut,
        }

    def _relax_mutant(self, coords: torch.Tensor, mut_seq: str, mut_global_pos: int) -> Tuple[torch.Tensor, float]:
        """Local relaxation around the mutation site (window ±3 residues)."""
        L = len(mut_seq)
        window_start = max(0, mut_global_pos - 3)
        window_end = min(L, mut_global_pos + 4)
        # Create optimizable tensor for the window only (but we need full coords for energy)
        # We'll optimize full coords but mask gradients outside window.
        coords_opt = coords.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([coords_opt], lr=1e-3)
        best_E = float('inf')
        best_coords = coords_opt.clone()
        chain_types_per_res = []
        for seq, ct in zip(self.sequences, self.chain_types):
            chain_types_per_res.extend([ct] * len(seq))
        alpha = self.alpha
        for _ in range(self.config.relaxation_steps):
            opt.zero_grad()
            ei, ed = sparse_edges(coords_opt, self.v30_cfg.sparse_cutoff, self.v30_cfg.max_neighbors)
            atoms = reconstruct_backbone(coords_opt)
            ei_hb, ed_hb = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.v30_cfg.max_neighbors)
            chi = self.chi  # keep sidechain fixed for speed (can be extended)
            E = total_physics_energy(coords_opt, mut_seq, alpha, chi,
                                     ei, ed, ei_hb, ed_hb,
                                     [len(s) for s in self.sequences[:-1]] if len(self.sequences)>1 else [],
                                     self.v30_cfg,
                                     chain_types=chain_types_per_res)
            E.backward()
            # Zero gradient outside window
            if coords_opt.grad is not None:
                mask = torch.zeros(L, device=coords_opt.device)
                mask[window_start:window_end] = 1.0
                coords_opt.grad *= mask.unsqueeze(-1)
            opt.step()
            if E.item() < best_E:
                best_E = E.item()
                best_coords = coords_opt.clone()
        return best_coords.detach(), best_E

    def _compute_energy(self, coords: torch.Tensor, seq: str) -> float:
        alpha = self.alpha
        ei, ed = sparse_edges(coords, self.v30_cfg.sparse_cutoff, self.v30_cfg.max_neighbors)
        atoms = reconstruct_backbone(coords)
        ei_hb, ed_hb = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.v30_cfg.max_neighbors)
        chi = self.chi
        chain_types_per_res = []
        for s, ct in zip(self.sequences, self.chain_types):
            chain_types_per_res.extend([ct] * len(s))
        with torch.no_grad():
            E = total_physics_energy(coords, seq, alpha, chi,
                                     ei, ed, ei_hb, ed_hb,
                                     [len(s) for s in self.sequences[:-1]] if len(self.sequences)>1 else [],
                                     self.v30_cfg,
                                     chain_types=chain_types_per_res)
        return E.item()

    def scan_all_single(self):
        """Full single‑mutation scan across all chains."""
        results = []
        for chain_idx, (seq, ct) in enumerate(zip(self.sequences, self.chain_types)):
            alphabet = get_alphabet(seq)
            logger.info(f"Chain {chain_idx} ({ct}) length {len(seq)}, alphabet size {len(alphabet)}")
            for pos in tqdm(range(len(seq)), desc=f"Chain {chain_idx} mutations"):
                wt = seq[pos]
                for mut in alphabet:
                    if mut == wt: continue
                    res = self.compute_ddg_single(chain_idx, pos, mut, relax=(self.config.relaxation_steps>0))
                    results.append(res)
        return results

    def scan_epistasis(self, pairs: Optional[List[Tuple[int,int,int,int]]] = None):
        """Scan double mutations. pairs: list of (chain1, pos1, chain2, pos2)."""
        if pairs is None:
            # Auto‑generate limited set (for demo)
            pairs = []
            for c in range(len(self.sequences)):
                seq = self.sequences[c]
                alphabet = get_alphabet(seq)
                for i in range(len(seq)):
                    for j in range(i+2, len(seq)):
                        # limit
                        if len(pairs) >= 200: break
                        pairs.append((c, i, c, j))
            logger.info(f"Auto‑generated {len(pairs)} epistasis pairs.")
        results = []
        for c1, p1, c2, p2 in tqdm(pairs, desc="Epistasis pairs"):
            # Get single mutant ddG
            wt1 = self.sequences[c1][p1]
            wt2 = self.sequences[c2][p2]
            alphabet = get_alphabet(self.sequences[c1])
            for m1 in alphabet:
                if m1 == wt1: continue
                for m2 in alphabet:
                    if m2 == wt2: continue
                    d1 = self.compute_ddg_single(c1, p1, m1, relax=True)
                    d2 = self.compute_ddg_single(c2, p2, m2, relax=True)
                    # Double mutant
                    mut_seq = self.full_seq
                    offset1 = sum(len(s) for s in self.sequences[:c1])
                    offset2 = sum(len(s) for s in self.sequences[:c2])
                    gp1 = offset1 + p1
                    gp2 = offset2 + p2
                    mut_full = mut_seq[:gp1] + m1 + mut_seq[gp1+1:gp2] + m2 + mut_seq[gp2+1:]
                    coords = self.ca_coords.clone()
                    # Relax double mutant around both positions
                    if self.config.relaxation_steps > 0:
                        coords, e_dbl = self._relax_mutant_double(coords, mut_full, gp1, gp2)
                    else:
                        e_dbl = self._compute_energy(coords, mut_full)
                    ddg_dbl = e_dbl - self.wt_energy
                    additive = d1['ddg'] + d2['ddg']
                    eps = ddg_dbl - additive
                    results.append({
                        'chain1': c1, 'pos1': p1, 'mut1': m1, 'ddg1': d1['ddg'],
                        'chain2': c2, 'pos2': p2, 'mut2': m2, 'ddg2': d2['ddg'],
                        'ddg_double': ddg_dbl,
                        'ddg_additive': additive,
                        'epistasis': eps,
                        'significant': abs(eps) > self.config.ddg_threshold,
                    })
        return results

    def _relax_mutant_double(self, coords, mut_seq, gp1, gp2):
        L = len(mut_seq)
        window_start = min(gp1, gp2) - 3
        window_end = max(gp1, gp2) + 4
        window_start = max(0, window_start)
        window_end = min(L, window_end)
        coords_opt = coords.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([coords_opt], lr=1e-3)
        best_E = float('inf')
        best_coords = coords_opt.clone()
        alpha = self.alpha
        chain_types_per_res = []
        for s, ct in zip(self.sequences, self.chain_types):
            chain_types_per_res.extend([ct]*len(s))
        for _ in range(self.config.relaxation_steps):
            opt.zero_grad()
            ei, ed = sparse_edges(coords_opt, self.v30_cfg.sparse_cutoff, self.v30_cfg.max_neighbors)
            atoms = reconstruct_backbone(coords_opt)
            ei_hb, ed_hb = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.v30_cfg.max_neighbors)
            chi = self.chi
            E = total_physics_energy(coords_opt, mut_seq, alpha, chi,
                                     ei, ed, ei_hb, ed_hb,
                                     [len(s) for s in self.sequences[:-1]] if len(self.sequences)>1 else [],
                                     self.v30_cfg, chain_types=chain_types_per_res)
            E.backward()
            if coords_opt.grad is not None:
                mask = torch.zeros(L, device=coords_opt.device)
                mask[window_start:window_end] = 1.0
                coords_opt.grad *= mask.unsqueeze(-1)
            opt.step()
            if E.item() < best_E:
                best_E = E.item()
                best_coords = coords_opt.clone()
        return best_coords.detach(), best_E

    def run_analysis(self):
        self.load_structure()
        results = {'wt_energy': self.wt_energy}
        if self.config.scan_full:
            logger.info("Full single‑mutation scan...")
            scan_res = self.scan_all_single()
            results['scan_results'] = scan_res
            self._generate_scan_reports(scan_res)
        if self.config.scan_epistasis or self.config.epistasis_pairs:
            logger.info("Epistasis scan...")
            epi_res = self.scan_epistasis(self.config.epistasis_pairs)
            results['epistasis_results'] = epi_res
            self._generate_epistasis_reports(epi_res)
        # Export summary JSON
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, "summary.json"), "w") as f:
            json.dump({k: v if not isinstance(v, list) else f"{len(v)} records" for k,v in results.items()}, f, indent=2)
        logger.info(f"Results saved in {self.config.output_dir}")

    # Plotting functions (similar to earlier HTS but unified)
    def _generate_scan_reports(self, results):
        out_dir = Path(self.config.output_dir)
        df = pd.DataFrame(results)
        df.to_csv(out_dir / "scan_ddg.csv", index=False)
        # ... (implement heatmap, profile, distributions using chain_types) ...
        # For brevity, we'll reuse code from v31.1 and v32.1 with auto coloring by type.
        pass

    def _generate_epistasis_reports(self, results):
        out_dir = Path(self.config.output_dir)
        df = pd.DataFrame(results)
        df.to_csv(out_dir / "epistasis.csv", index=False)
        # plots...
        pass


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTS FOLD v33 — Universal Mutation Scanner")
    parser.add_argument('--pdb', type=str, help='PDB file (can contain protein/DNA/RNA)')
    parser.add_argument('--seq', type=str, help='Sequence (auto‑detect type)')
    parser.add_argument('--output', default='./hts_v33_output')
    parser.add_argument('--scan', action='store_true', help='Full single mutation scan')
    parser.add_argument('--epistasis', action='store_true', help='Epistasis scan')
    parser.add_argument('--relax', type=int, default=30, help='Relaxation steps')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()

    config = HTSConfigV33(
        pdb_structure=args.pdb,
        sequence=args.seq,
        output_dir=args.output,
        scan_full=args.scan,
        scan_epistasis=args.epistasis,
        relaxation_steps=args.relax,
        use_gpu=args.gpu,
        num_gpus=args.num_gpus,
    )
    analyzer = HTSAnalyzerV33(config)
    analyzer.run_analysis()

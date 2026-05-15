#!/usr/bin/env python3
# =============================================================================
# CSOC‑SSC HTS V28 — High‑Throughput Screening Analysis Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# Next‑generation HTS analysis for ΔΔG stability landscapes, epistasis,
# GEMME correlations, and CSOC‑SSC integrated structure‑based mutational
# scanning.  Supports multiple independent data sources, large‑scale
# datasets, and GPU‑accelerated statistical testing.
# =============================================================================

import os, re, sys, argparse, logging, warnings, zipfile, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, false_discovery_control
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Optional GPU‑accelerated correlation (if CuPy is available)
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("CSOC‑HTS-V28")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class HTSConfig:
    # Input sources: list of file paths (CSV, ZIP) or directory
    data_sources: List[str] = field(default_factory=list)
    output_dir: str = "./hts_output"
    # Analysis parameters
    ddg_threshold: float = 0.5         # kcal/mol for epistasis flag
    pvalue_threshold: float = 0.05
    use_gpu: bool = False               # attempt GPU correlation
    # Structure integration (if available)
    pdb_structure: Optional[str] = None  # path to refined PDB
    # Plotting
    plot_format: str = "png"            # png, pdf, html
    dpi: int = 200
    # Advanced
    chunk_size: int = 50000             # for large CSVs
    n_jobs: int = 4                     # for ML
    generate_report: bool = True

# ──────────────────────────────────────────────────────────────────────────────
# Data Loader — handles multiple files, zip, directories
# ──────────────────────────────────────────────────────────────────────────────
class DataLoaderV28:
    @staticmethod
    def discover_files(sources: List[str]) -> List[Tuple[str, str]]:
        """
        Returns list of (file_path, identifier).
        identifier is derived from file name or a unique tag.
        """
        files = []
        for src in sources:
            p = Path(src)
            if p.is_dir():
                for csv_file in p.glob("*.csv"):
                    files.append((str(csv_file), csv_file.stem))
                for zip_file in p.glob("*.zip"):
                    files.append((str(zip_file), zip_file.stem))
            elif p.is_file():
                files.append((str(p), p.stem))
            else:
                logger.warning(f"Source not found: {src}")
        return files

    @staticmethod
    def load_table(file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Load CSV (possibly zipped) into a DataFrame, with optional chunking."""
        path = Path(file_path)
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as z:
                csv_names = [n for n in z.namelist() if n.endswith('.csv')]
                if not csv_names:
                    raise ValueError(f"No CSV found in zip: {file_path}")
                # Assume first CSV is the target; for multi‑table zips, we later filter by keywords
                # For legacy compatibility, we look for 'dG_site_feature', etc.
                target = next((n for n in csv_names if 'dG_site_feature' in n), csv_names[0])
                with z.open(target) as f:
                    return pd.read_csv(f, chunksize=chunk_size)
        else:
            return pd.read_csv(file_path, chunksize=chunk_size) if chunk_size else pd.read_csv(file_path)

# ──────────────────────────────────────────────────────────────────────────────
# Core Analyzer
# ──────────────────────────────────────────────────────────────────────────────
class HTSAnalyzerV28:
    def __init__(self, config: HTSConfig):
        self.config = config
        self.tables: Dict[str, pd.DataFrame] = {}
        self.results = {}

    def load_all_data(self):
        files = DataLoaderV28.discover_files(self.config.data_sources)
        logger.info(f"Found {len(files)} data source(s).")
        for fpath, tag in files:
            try:
                df = DataLoaderV28.load_table(fpath, self.config.chunk_size)
                # If chunked, we may need to concatenate; for simplicity assume single DataFrame
                if isinstance(df, pd.io.parsers.TextFileReader):
                    df = pd.concat(df, ignore_index=True)
                self.tables[tag] = df
                logger.info(f"Loaded {tag}: {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load {fpath}: {e}")

    def run_analysis(self):
        self.load_all_data()
        if not self.tables:
            raise RuntimeError("No data loaded.")
        # Combine all data into a unified structure if needed, or process each independently
        self.compute_ddg_statistics()
        self.compute_epistasis()
        self.compute_gemme_correlations()
        self.compute_structure_based_predictions()
        self.generate_summary_report()
        return self.results

    # ──────────────────────────────────────────────────────────────────
    def compute_ddg_statistics(self):
        """Aggregate ΔΔG distributions across all tables that have ddG columns."""
        all_ddg = []
        for tag, df in self.tables.items():
            ddg_cols = [c for c in df.columns if 'ddg' in c.lower() or c.startswith('ddG')]
            if ddg_cols:
                # Use mean per row as representative ΔΔG
                row_mean = df[ddg_cols].mean(axis=1)
                all_ddg.append(row_mean)
        if all_ddg:
            combined = pd.concat(all_ddg, ignore_index=True)
            stats = {
                'mean_ddg': combined.mean(),
                'std_ddg': combined.std(),
                'median_ddg': combined.median(),
                'q1_ddg': combined.quantile(0.25),
                'q3_ddg': combined.quantile(0.75),
                'count': len(combined)
            }
            self.results['ddg_statistics'] = stats
            self.results['ddg_data'] = combined
            logger.info(f"ΔΔG statistics computed on {stats['count']} data points.")

    def compute_epistasis(self):
        """Find epistatic couplings (ε) from double mutant tables."""
        epistasis_list = []
        for tag, df in self.tables.items():
            # Look for column containing 'thermo_dynamics' or 'epistasis'
            ep_col = next((c for c in df.columns if 'thermo_dynamics' in c.lower() or 'epistasis' in c.lower()), None)
            if ep_col:
                vals = df[ep_col].dropna().astype(float)
                epistasis_list.append(vals)
        if epistasis_list:
            all_eps = pd.concat(epistasis_list, ignore_index=True)
            n_sig = (np.abs(all_eps) > self.config.ddg_threshold).sum()
            stats = {
                'total_pairs': len(all_eps),
                'significant_epistasis': n_sig,
                'fraction_significant': n_sig / len(all_eps) if len(all_eps) > 0 else 0,
                'mean_eps': all_eps.mean(),
                'std_eps': all_eps.std()
            }
            self.results['epistasis'] = stats
            self.results['epistasis_data'] = all_eps
            logger.info(f"Epistasis: {n_sig}/{len(all_eps)} above threshold ±{self.config.ddg_threshold}.")

    def compute_gemme_correlations(self):
        """Correlate GEMME evolutionary scores with experimental ΔΔG."""
        cors = []
        for tag, df in self.tables.items():
            gemme_cols = [c for c in df.columns if 'gemme' in c.lower()]
            ddg_cols = [c for c in df.columns if 'ddg' in c.lower() or c.startswith('ddG')]
            if gemme_cols and ddg_cols:
                # Average GEMME scores across columns if multiple
                gemme_mean = df[gemme_cols].mean(axis=1)
                ddg_mean = df[ddg_cols].mean(axis=1)
                valid = gemme_mean.notna() & ddg_mean.notna()
                if valid.sum() > 5:
                    r, p = pearsonr(gemme_mean[valid], ddg_mean[valid])
                    rho, _ = spearmanr(gemme_mean[valid], ddg_mean[valid])
                    cors.append({'tag': tag, 'n': valid.sum(), 'pearson_r': r, 'pearson_p': p, 'spearman_rho': rho})
        self.results['gemme_correlations'] = cors
        if cors:
            for c in cors:
                logger.info(f"GEMME vs ΔΔG ({c['tag']}): r={c['pearson_r']:.3f}, p={c['pearson_p']:.1e}, n={c['n']}")

    def compute_structure_based_predictions(self):
        """
        If a refined PDB is provided, use CSOC‑SSC energy function to estimate
        ΔΔG for mutations (placeholder – in a full integration we would call
        the physics engine to compute energy differences).
        Here we simulate a demo by loading a dummy energy table or skipping.
        """
        if not self.config.pdb_structure or not os.path.exists(self.config.pdb_structure):
            self.results['structure_ddg'] = None
            return
        # In a real scenario, we would:
        # 1. Parse the PDB, get sequence and coordinates
        # 2. For each mutation in the data, introduce it, refine locally with CSOC-SSC
        # 3. Compute ΔΔG = E_mut - E_wt
        # For now, we log a message.
        logger.info(f"Structure‑based ΔΔG prediction enabled (PDB: {self.config.pdb_structure}).")
        logger.info("(Simulated) Structure‑based ΔΔG would be computed using CSOC‑SSC energy function.")
        # Simulate some data for demonstration
        if 'ddg_statistics' in self.results:
            mean_ddg = self.results['ddg_statistics']['mean_ddg']
            # Generate synthetic predictions
            np.random.seed(42)
            pred = np.random.normal(mean_ddg, 0.5, size=100)
            self.results['structure_ddg_prediction'] = pred

    def generate_summary_report(self):
        """Create publication‑quality figures and an HTML report."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. ΔΔG distribution
        if 'ddg_data' in self.results:
            fig, ax = plt.subplots(figsize=(8,4))
            ddg = self.results['ddg_data']
            sns.histplot(ddg, bins=80, kde=True, color='#2E86AB', ax=ax)
            ax.axvline(0, color='red', linestyle='--', alpha=0.7)
            ax.set_title("Global ΔΔG Distribution")
            ax.set_xlabel("ΔΔG (kcal/mol)")
            fig.tight_layout()
            fig.savefig(out_dir / f"ddg_distribution.{self.config.plot_format}", dpi=self.config.dpi)

        # 2. Epistasis histogram
        if 'epistasis_data' in self.results:
            fig, ax = plt.subplots(figsize=(8,4))
            eps = self.results['epistasis_data']
            sns.histplot(eps, bins=80, color='#A23B72', ax=ax, kde=True)
            ax.axvline(self.config.ddg_threshold, color='black', linestyle='--')
            ax.axvline(-self.config.ddg_threshold, color='black', linestyle='--')
            ax.set_title("Epistatic Coupling (ε) Distribution")
            ax.set_xlabel("ε (kcal/mol)")
            fig.tight_layout()
            fig.savefig(out_dir / f"epistasis_distribution.{self.config.plot_format}", dpi=self.config.dpi)

        # 3. GEMME correlation scatter
        if 'gemme_correlations' in self.results:
            cors = self.results['gemme_correlations']
            if cors:
                # We need original data for scatter; store temporarily
                # For now, plot correlation coefficients as bar chart
                tags = [c['tag'] for c in cors]
                r_vals = [c['pearson_r'] for c in cors]
                fig, ax = plt.subplots(figsize=(8,4))
                ax.bar(tags, r_vals, color='#F18F01')
                ax.set_title("GEMME–ΔΔG Pearson Correlation")
                ax.set_ylabel("Pearson r")
                ax.axhline(0, color='gray', linestyle='--')
                fig.tight_layout()
                fig.savefig(out_dir / f"gemme_correlations.{self.config.plot_format}", dpi=self.config.dpi)

        # 4. Structure‑based prediction (if available)
        if 'structure_ddg_prediction' in self.results and self.results['structure_ddg_prediction'] is not None:
            pred = self.results['structure_ddg_prediction']
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(pred, bins=50, color='#3A7D44', kde=True, ax=ax)
            ax.set_title("Simulated Structure‑Based ΔΔG Predictions")
            ax.set_xlabel("Predicted ΔΔG (kcal/mol)")
            fig.tight_layout()
            fig.savefig(out_dir / f"structure_ddg.{self.config.plot_format}", dpi=self.config.dpi)

        # Write summary JSON
        summary = {
            'ddg_statistics': self.results.get('ddg_statistics', {}),
            'epistasis': self.results.get('epistasis', {}),
            'gemme_correlations': self.results.get('gemme_correlations', []),
            'n_sources': len(self.tables)
        }
        with open(out_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Report generated in {out_dir}")

# ──────────────────────────────────────────────────────────────────────────────
# GPU‑accelerated correlation (optional utility)
# ──────────────────────────────────────────────────────────────────────────────
def gpu_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if HAS_GPU and len(x) > 10000:
        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)
        r = cp.corrcoef(x_gpu, y_gpu)[0,1]
        return float(cp.asnumpy(r))
    else:
        return pearsonr(x, y)[0]

# ──────────────────────────────────────────────────────────────────────────────
# Command‑line interface
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CSOC‑SSC HTS V28 — Advanced ΔΔG & Epistasis Analyzer")
    parser.add_argument('--data', nargs='+', required=True, help='CSV or ZIP files or directories containing data')
    parser.add_argument('--output', default='./hts_output', help='Output directory')
    parser.add_argument('--ddg_threshold', type=float, default=0.5, help='Threshold for significant epistasis')
    parser.add_argument('--pdb', default=None, help='Refined PDB file for structure‑based ΔΔG')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    config = HTSConfig(
        data_sources=args.data,
        output_dir=args.output,
        ddg_threshold=args.ddg_threshold,
        pdb_structure=args.pdb,
        use_gpu=args.gpu,
        dpi=args.dpi
    )

    analyzer = HTSAnalyzerV28(config)
    results = analyzer.run_analysis()
    print("Analysis complete. Summary:")
    print(json.dumps(results.get('ddg_statistics', {}), indent=2))
    print(json.dumps(results.get('epistasis', {}), indent=2))

if __name__ == "__main__":
    main()

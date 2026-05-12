"""
Visualize ConditionalVAE predictions vs ground truth on the tokamak grid.

Layout (overview):
  Row 0 — GT             : te | ti | na_D1 | ua_D1   (LogNorm / SymLogNorm)
  Row 1 — Pred (GT scale): same norm as GT → magnitude comparison
  Row 2 — Pred (own scale): rescaled to pred range → spatial structure visible
  Row 3 — |error| [%]   : relative absolute error, LogNorm

Usage:
    python visualize_physics_model.py                         # random test sample
    python visualize_physics_model.py --sample_idx 42
    python visualize_physics_model.py --checkpoint physics_model/checkpoints/best.pt
    python visualize_physics_model.py --field te              # detailed 4-col view
    python visualize_physics_model.py --species 0 1 2         # species for na/ua
    python visualize_physics_model.py --show_norm             # also show normalised heatmap
    python visualize_physics_model.py --save out.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
import numpy as np
import torch

from data_handler import PlasmaDataHandler
from physics_model.model import ConditionalVAE


# ---------------------------------------------------------------------------
# Denormalizer (numpy, for visualization only)
# ---------------------------------------------------------------------------

class Denormalizer:
    """Reverses min-max (and optional natural-log) normalization."""

    def __init__(self, norm_stats: dict):
        self.stats = norm_stats

    def __call__(self, field: str, data: np.ndarray) -> np.ndarray:
        vmin = self.stats[f"{field}_min"].astype(np.float64)
        vmax = self.stats[f"{field}_max"].astype(np.float64)

        # After recalc_norm_stats fix: te/ti/na store LOG values in vmin/vmax
        # These fields need exp() to denormalize back to physical space
        if field in ("te", "ti", "na"):
            # vmin/vmax are already in log space, so denormalize in log space then exp
            log_val = data * (vmax - vmin) + vmin
            return np.exp(np.clip(log_val, vmin, vmax))
        
        # ua: arcsinh transform (handles bipolar velocities)
        elif field == "ua":
            # vmin/vmax are arrays shape (10,) for species
            scale = np.maximum(np.abs(vmin), np.abs(vmax))
            scale = np.maximum(scale, 1.0)
            asinh_max = np.arcsinh(1.0)
            # Reverse: [0,1] → [-asinh_max, asinh_max] → sinh → original scale
            data_arcsinh = data * (2.0 * asinh_max) - asinh_max
            return scale * np.sinh(data_arcsinh)
        
        # X and other linear fields
        else:
            return data * (vmax - vmin) + vmin


# ---------------------------------------------------------------------------
# Colormap normalization helpers
# ---------------------------------------------------------------------------

def _make_norm(field: str, values: np.ndarray) -> mcolors.Normalize:
    """
    Appropriate matplotlib norm for a physical field:
      te, ti, na → LogNorm  (many decades; vacuum cells near-zero floored)
      ua         → SymLogNorm  (bipolar)
      others     → linear Normalize
    """
    finite = values.ravel()[np.isfinite(values.ravel())]
    if len(finite) == 0:
        return mcolors.Normalize(vmin=0, vmax=1)

    if field in ("te", "ti", "na"):
        pos  = finite[finite > 0]
        if len(pos) == 0:
            return mcolors.Normalize(vmin=0, vmax=1)
        vmin = np.percentile(pos, 2)
        vmax = np.percentile(finite, 98)
        # floor at 1e-8 of vmax so vacuum cells don't crush the colour scale
        return mcolors.LogNorm(vmin=max(vmin, vmax * 1e-8), vmax=max(vmax, 1e-30))

    elif field == "ua":
        absmax = np.percentile(np.abs(finite), 98)
        if absmax == 0:
            absmax = 1.0
        linthresh = max(absmax * 1e-2, 1.0)
        return mcolors.SymLogNorm(
            linthresh=linthresh, vmin=-absmax, vmax=absmax, base=10
        )

    vmin, vmax = np.percentile(finite, [2, 98])
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _make_err_norm(rel_err: np.ndarray) -> mcolors.LogNorm:
    """LogNorm for relative-error [%] fields."""
    finite = rel_err.ravel()
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if len(finite) == 0:
        return mcolors.LogNorm(vmin=0.1, vmax=100)
    vmin = max(np.percentile(finite, 5), 0.01)
    vmax = min(np.percentile(finite, 95), 1000)
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)



# ---------------------------------------------------------------------------
# Tokamak grid plotting
# ---------------------------------------------------------------------------

def _cell_polys(crx: np.ndarray, cry: np.ndarray) -> np.ndarray:
    """(nx, ny, 4) corner arrays → (nx*ny, 4, 2) polygon vertices."""
    verts = np.stack([crx, cry], axis=-1)   # (nx, ny, 4, 2)
    return verts.reshape(-1, 4, 2)


def plot_field(
    ax: plt.Axes,
    crx: np.ndarray,
    cry: np.ndarray,
    field_2d: np.ndarray,
    title: str = "",
    cmap: str = "plasma",
    norm: mcolors.Normalize = None,
) -> plt.cm.ScalarMappable:
    """
    Draw field_2d (nx, ny) on the irregular tokamak quad-mesh.
    Returns ScalarMappable for shared colorbar control.
    """
    verts  = _cell_polys(crx, cry)
    values = field_2d.ravel()

    if norm is None:
        vmin, vmax = np.nanpercentile(values, [2, 98])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    col = PolyCollection(
        verts,
        array=values,
        cmap=plt.get_cmap(cmap),
        norm=norm,
        linewidths=0,
        antialiaseds=False,
    )
    ax.add_collection(col)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]", fontsize=7)
    ax.set_ylabel("Z [m]", fontsize=7)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
    sm.set_array([])
    return sm


def _add_cbar(fig, sm, ax, label="", log=False):
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=5)
    # Improved formatting for scientific notation
    if log:
        cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
    else:
        # Force scientific notation for very small/large numbers
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        cb.ax.yaxis.set_major_formatter(formatter)


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

SPECIES_NAMES = ["D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
FIELD_LABELS  = {"te": "Te [J]", "ti": "Ti [J]", "na": "n [m⁻³]", "ua": "u [m/s]"}
FIELD_CMAPS   = {"te": "inferno", "ti": "inferno", "na": "viridis", "ua": "RdBu_r"}

def load_model(checkpoint_path: str, device: torch.device) -> ConditionalVAE:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_d = ckpt.get("args", {})
    model = ConditionalVAE(
        in_channels=22,
        out_channels=22,
        latent_dim=args_d.get("latent_dim", 128),
        cond_dim=8,
        nx=104,
        ny=50,
        use_prior_net=args_d.get("use_prior_net", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    best  = ckpt.get("best_val_recon", ckpt.get("best_val_loss", float("inf")))
    print(f"Checkpoint: epoch={epoch}, best_val_recon={best:.4f}")
    return model


def get_sample(handler: PlasmaDataHandler, idx: int):
    """Normalise and stack one sample → (1,22,104,50), (1,8)."""
    h = handler
    te = h.normalize("te", h.te[idx][None])[0]
    ti = h.normalize("ti", h.ti[idx][None])[0]
    na = h.normalize("na", h.na[idx][None])[0]
    ua = h.normalize("ua", h.ua[idx][None])[0]
    X  = h.normalize("X",  h.X[idx][None])[0]

    te_t = torch.tensor(te, dtype=torch.float32).unsqueeze(0)
    ti_t = torch.tensor(ti, dtype=torch.float32).unsqueeze(0)
    na_t = torch.tensor(na, dtype=torch.float32).permute(2, 0, 1)
    ua_t = torch.tensor(ua, dtype=torch.float32).permute(2, 0, 1)
    fields = torch.cat([te_t, ti_t, na_t, ua_t], dim=0).unsqueeze(0)
    cond   = torch.tensor(X,  dtype=torch.float32).unsqueeze(0)
    return fields, cond


def unpack_norm(x: np.ndarray):
    """(22,104,50) → te(104,50), ti(104,50), na(104,50,10), ua(104,50,10)."""
    te = x[0]
    ti = x[1]
    na = x[2:12].transpose(1, 2, 0)
    ua = x[12:22].transpose(1, 2, 0)
    return te, ti, na, ua


# ---------------------------------------------------------------------------
# Console diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(gt_norm, pred_norm, denorm):
    print("\n" + "=" * 80)
    print("DIAGNOSTICS  (normalised space  — expected range [0, 1])")
    print(f"  {'head':>4s}  {'GT range':>18s}  {'Pred range':>18s}  {'RMSE':>8s}  {'corr':>7s}")
    print("-" * 80)
    for fname, sl in [("te", slice(0, 1)), ("ti", slice(1, 2)),
                      ("na", slice(2, 12)), ("ua", slice(12, 22))]:
        g  = gt_norm[sl]
        p  = pred_norm[sl]
        rmse = np.sqrt(np.mean((p - g) ** 2))
        corr = float(np.corrcoef(p.ravel(), g.ravel())[0, 1])
        print(
            f"  {fname:>4s}  "
            f"[{g.min():+.3f}, {g.max():+.3f}]  "
            f"[{p.min():+.3f}, {p.max():+.3f}]  "
            f"{rmse:8.4f}  {corr:+.4f}"
        )

    print("\nPHYSICAL SPACE  (after denorm)")
    print("-" * 75)
    for fname, fi, unit in [("te", 0, "J"), ("ti", 1, "J"), ("na", 2, "m⁻³"), ("ua", 3, "m/s")]:
        gt_p   = denorm(fname, unpack_norm(gt_norm)[fi])
        pred_p = denorm(fname, unpack_norm(pred_norm)[fi])
        pg  = np.nanpercentile(gt_p,   [2, 50, 98])
        pp  = np.nanpercentile(pred_p, [2, 50, 98])
        print(
            f"  {fname:3s}  GT  p2={pg[0]:.3e}  median={pg[1]:.3e}  p98={pg[2]:.3e}  [{unit}]"
        )
        print(
            f"       Pred p2={pp[0]:.3e}  median={pp[1]:.3e}  p98={pp[2]:.3e}  [{unit}]"
        )
    print("=" * 75 + "\n")


# ---------------------------------------------------------------------------
# Overview figure  (4 rows × 4 columns)
# ---------------------------------------------------------------------------

OVERVIEW_PANELS = [
    ("te", -1, "inferno"),
    ("ti", -1, "inferno"),
    ("na",  1, "viridis"),   # D1 species
    ("ua",  1, "RdBu_r"),
]


def make_overview_figure(crx, cry, denorm, gt_norm, pred_norm, sample_idx):
    """
    4 rows × 4 columns:
      Row 0  GT               (LogNorm / SymLogNorm)
      Row 1  Pred – GT scale  (same norm → magnitude comparison)
      Row 2  Pred – own scale (rescaled → spatial structure visible)
      Row 3  |Pred − GT| / |GT| × 100 %  (LogNorm)
    """
    field_map = ["te", "ti", "na", "ua"]

    fig, axes = plt.subplots(4, 4, figsize=(20, 18), squeeze=False)
    fig.suptitle(
        f"Physics model — sample idx={sample_idx}\n"
        "Row 0: GT  |  Row 1: Pred (GT scale)  "
        "|  Row 2: Pred (own scale)  |  Row 3: |err| %",
        fontsize=12, fontweight="bold",
    )
    for row, lbl in enumerate(["GT", "Pred\n(GT scale)",
                                "Pred\n(own scale)", "|err| %"]):
        axes[row, 0].set_ylabel(lbl, fontsize=9, labelpad=4)

    for col, (fld, sp, cmap) in enumerate(OVERVIEW_PANELS):
        fi = field_map.index(fld)
        gt_phys   = denorm(fld, unpack_norm(gt_norm)[fi])
        pred_phys = denorm(fld, unpack_norm(pred_norm)[fi])
        gt_2d   = gt_phys   if sp < 0 else gt_phys[..., sp]
        pred_2d = pred_phys if sp < 0 else pred_phys[..., sp]

        sp_name = "" if sp < 0 else f" {SPECIES_NAMES[sp]}"
        label   = FIELD_LABELS[fld] + sp_name

        # GT  (LogNorm)
        gt_norm_obj = _make_norm(fld, gt_2d)
        is_log = isinstance(gt_norm_obj, (mcolors.LogNorm, mcolors.SymLogNorm))
        sm0 = plot_field(axes[0, col], crx, cry, gt_2d,
                         title=f"GT  {label}", cmap=cmap, norm=gt_norm_obj)
        _add_cbar(fig, sm0, axes[0, col], label=label, log=is_log)

        # Pred – GT scale
        sm1 = plot_field(axes[1, col], crx, cry, pred_2d,
                         title=f"Pred (GT scale)  {label}", cmap=cmap,
                         norm=gt_norm_obj)
        _add_cbar(fig, sm1, axes[1, col], label=label, log=is_log)

        # Pred – own scale  (reveals spatial structure even if magnitude is off)
        pred_norm_obj = _make_norm(fld, pred_2d)
        sm2 = plot_field(axes[2, col], crx, cry, pred_2d,
                         title=f"Pred (own scale)  {label}", cmap=cmap,
                         norm=pred_norm_obj)
        _add_cbar(fig, sm2, axes[2, col], label=label,
                  log=isinstance(pred_norm_obj, (mcolors.LogNorm, mcolors.SymLogNorm)))

        # Relative error [%]
        gt_scale = np.nanpercentile(np.abs(gt_2d[gt_2d != 0]), 50) if np.any(gt_2d != 0) else 1.0
        denom    = np.where(np.abs(gt_2d) < gt_scale * 1e-6, gt_scale * 1e-6, np.abs(gt_2d))
        rel_err  = np.abs(pred_2d - gt_2d) / denom * 100.0
        sm3 = plot_field(axes[3, col], crx, cry, rel_err,
                         title=f"|err| %  {label}", cmap="hot_r",
                         norm=_make_err_norm(rel_err))
        _add_cbar(fig, sm3, axes[3, col], label="%", log=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Detailed per-field comparison (4 columns: GT | Pred GT-scale | Pred own | err)
# ---------------------------------------------------------------------------

def make_comparison_figure(
    crx, cry, denorm,
    gt_norm, pred_norm,
    field, species_indices, sample_idx,
):
    is_scalar = field in ("te", "ti")
    n_rows = 1 if is_scalar else len(species_indices)

    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows + 0.5), squeeze=False)
    sp_str = "scalar" if is_scalar else str([SPECIES_NAMES[i] for i in species_indices])
    fig.suptitle(
        f"Field: {field.upper()}  |  Sample idx={sample_idx}  |  {sp_str}\n"
        "GT  |  Pred (GT scale)  |  Pred (own scale)  |  |err| %",
        fontsize=11, fontweight="bold",
    )

    fi = ["te", "ti", "na", "ua"].index(field)
    gt_phys_full   = denorm(field, unpack_norm(gt_norm)[fi])
    pred_phys_full = denorm(field, unpack_norm(pred_norm)[fi])
    cmap = FIELD_CMAPS[field]

    def _slice(phys, sp):
        return phys if is_scalar else phys[..., sp]

    for row_i, sp in enumerate([-1] if is_scalar else species_indices):
        gt_2d   = _slice(gt_phys_full,   sp)
        pred_2d = _slice(pred_phys_full, sp)
        sp_name = "" if is_scalar else f"  {SPECIES_NAMES[sp]}"

        gt_norm_obj   = _make_norm(field, gt_2d)
        pred_norm_obj = _make_norm(field, pred_2d)
        is_log = isinstance(gt_norm_obj, (mcolors.LogNorm, mcolors.SymLogNorm))

        sm0 = plot_field(axes[row_i, 0], crx, cry, gt_2d,
                         title=f"GT  {FIELD_LABELS[field]}{sp_name}",
                         cmap=cmap, norm=gt_norm_obj)
        _add_cbar(fig, sm0, axes[row_i, 0], label=FIELD_LABELS[field], log=is_log)

        sm1 = plot_field(axes[row_i, 1], crx, cry, pred_2d,
                         title=f"Pred (GT scale){sp_name}",
                         cmap=cmap, norm=gt_norm_obj)
        _add_cbar(fig, sm1, axes[row_i, 1], label=FIELD_LABELS[field], log=is_log)

        sm2 = plot_field(axes[row_i, 2], crx, cry, pred_2d,
                         title=f"Pred (own scale){sp_name}",
                         cmap=cmap, norm=pred_norm_obj)
        _add_cbar(fig, sm2, axes[row_i, 2], label=FIELD_LABELS[field],
                  log=isinstance(pred_norm_obj, (mcolors.LogNorm, mcolors.SymLogNorm)))

        gt_scale = np.nanpercentile(np.abs(gt_2d[gt_2d != 0]), 50) if np.any(gt_2d != 0) else 1.0
        denom    = np.where(np.abs(gt_2d) < gt_scale * 1e-6, gt_scale * 1e-6, np.abs(gt_2d))
        rel_err  = np.abs(pred_2d - gt_2d) / denom * 100.0
        sm3 = plot_field(axes[row_i, 3], crx, cry, rel_err,
                         title=f"|err| %{sp_name}", cmap="hot_r",
                         norm=_make_err_norm(rel_err))
        _add_cbar(fig, sm3, axes[row_i, 3], label="%", log=True)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Normalised-space heatmap (debug: no denorm, raw model channels)
# ---------------------------------------------------------------------------

def make_norm_space_figure(gt_norm, pred_norm, sample_idx):
    """
    Normalised-space heatmaps for all 22 channels, grouped by decoder head.

    Layout: 6 rows × 10 cols
      rows 0-1  : te (col 0) and ti (col 1)   — scalar heads
      rows 2-3  : na species 0-9               — na head
      rows 4-5  : ua species 0-9               — ua head
    Each pair of rows is (GT, Pred). Cmaps match physical meaning:
      te/ti → inferno,  na → viridis,  ua → RdBu_r (centred on 0.5 = zero vel)
    """
    fig, axes = plt.subplots(6, 10, figsize=(24, 12), squeeze=False)
    fig.suptitle(
        f"Normalised space — per decoder head — sample idx={sample_idx}\n"
        "rows 0/2/4: GT  ·  rows 1/3/5: Pred  (expected ≈ [0, 1])",
        fontsize=11, fontweight="bold",
    )

    section_labels = [
        "te / ti  (GT)", "te / ti  (Pred)",
        "na head  (GT)", "na head  (Pred)",
        "ua head  (GT)", "ua head  (Pred)",
    ]
    for r, lbl in enumerate(section_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    def _show(ax, data, norm, cmap):
        ax.imshow(data, aspect="auto", origin="lower", norm=norm, cmap=cmap)
        ax.axis("off")

    # ── te (ch 0)  and  ti (ch 1) ─────────────────────────────────────────
    for col, (ch, name) in enumerate([(0, "te"), (1, "ti")]):
        vmin = min(gt_norm[ch].min(), pred_norm[ch].min())
        vmax = max(gt_norm[ch].max(), pred_norm[ch].max())
        n = mcolors.Normalize(vmin=vmin, vmax=vmax)
        _show(axes[0, col], gt_norm[ch],   n, "inferno")
        _show(axes[1, col], pred_norm[ch], n, "inferno")
        axes[0, col].set_title(name, fontsize=7)
    for col in range(2, 10):
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    # ── na head (ch 2–11) ─────────────────────────────────────────────────
    for sp in range(10):
        ch = 2 + sp
        vmin = min(gt_norm[ch].min(), pred_norm[ch].min())
        vmax = max(gt_norm[ch].max(), pred_norm[ch].max())
        n = mcolors.Normalize(vmin=vmin, vmax=vmax)
        _show(axes[2, sp], gt_norm[ch],   n, "viridis")
        _show(axes[3, sp], pred_norm[ch], n, "viridis")
        axes[2, sp].set_title(f"na {SPECIES_NAMES[sp]}", fontsize=6)

    # ── ua head (ch 12–21) ────────────────────────────────────────────────
    for sp in range(10):
        ch = 12 + sp
        vmin = min(gt_norm[ch].min(), pred_norm[ch].min())
        vmax = max(gt_norm[ch].max(), pred_norm[ch].max())
        # Centre RdBu on 0.5  (= zero velocity in normalised space)
        absdev = max(abs(vmin - 0.5), abs(vmax - 0.5), 1e-6)
        n = mcolors.Normalize(vmin=0.5 - absdev, vmax=0.5 + absdev)
        _show(axes[4, sp], gt_norm[ch],   n, "RdBu_r")
        _show(axes[5, sp], pred_norm[ch], n, "RdBu_r")
        axes[4, sp].set_title(f"ua {SPECIES_NAMES[sp]}", fontsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize ConditionalVAE predictions")
    p.add_argument("--checkpoint",  default="physics_model/checkpoints/best.pt")
    p.add_argument("--data_root",   default="a_dataset")
    p.add_argument("--split",       default="test", choices=["train", "test"])
    p.add_argument("--sample_idx",  type=int, default=None)
    p.add_argument("--field",       default="overview",
                   choices=["overview", "te", "ti", "na", "ua"])
    p.add_argument("--species",     type=int, nargs="+", default=[0, 1],
                   help="Species indices (0-9) for na/ua detailed view")
    p.add_argument("--use_encoder", action="store_true",
                   help="Encode the test input instead of sampling from p(z|c)")
    p.add_argument("--show_norm",   action="store_true",
                   help="Also show raw normalised-space heatmap")
    p.add_argument("--save",        default=None,
                   help="Save path (tag appended when multiple figures)")
    p.add_argument("--no_cuda",     action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    matplotlib.rcParams["axes.formatter.useoffset"] = False

    # ── Data ──────────────────────────────────────────────────────────────
    handler = PlasmaDataHandler(data_root=args.data_root)
    handler.load_geometry()
    handler.load_normalization_stats()
    handler.load_split(args.split)

    crx = handler.crx
    cry = handler.cry
    n   = handler.split_size
    idx = args.sample_idx if args.sample_idx is not None else np.random.randint(0, n)
    idx = int(np.clip(idx, 0, n - 1))
    print(f"Split='{args.split}'  sample idx={idx}  (total={n})")

    fields_t, cond_t = get_sample(handler, idx)
    fields_t = fields_t.to(device)
    cond_t   = cond_t.to(device)

    # ── Model ─────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    model = load_model(str(ckpt_path), device)

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        if args.use_encoder:
            pred_t, *_ = model(fields_t, cond_t)
            print("Inference: posterior (encoder + reparameterization)")
        else:
            pred_t = model.sample(num_samples=1, c=cond_t, use_prior=True)
            print("Inference: prior sampling p(z|c)  [no encoder]")

    gt_norm   = fields_t[0].cpu().numpy()
    pred_norm = pred_t[0].cpu().numpy()
    denorm    = Denormalizer(handler.norm_stats)

    # ── Diagnostics ───────────────────────────────────────────────────────
    print_diagnostics(gt_norm, pred_norm, denorm)

    # ── Figures ───────────────────────────────────────────────────────────
    figs = []
    if args.field == "overview":
        figs.append(("overview", make_overview_figure(
            crx, cry, denorm, gt_norm, pred_norm, idx
        )))
    else:
        figs.append((args.field, make_comparison_figure(
            crx, cry, denorm, gt_norm, pred_norm,
            field=args.field, species_indices=args.species, sample_idx=idx,
        )))

    if args.show_norm:
        figs.append(("norm_space", make_norm_space_figure(gt_norm, pred_norm, idx)))

    if args.save:
        base = Path(args.save)
        for tag, fig in figs:
            out = base.with_name(f"{base.stem}_{tag}{base.suffix}")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

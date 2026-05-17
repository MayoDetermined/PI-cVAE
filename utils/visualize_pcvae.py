"""
Visualise Parameter-Conditional VAE outputs vs ground truth on the tokamak curvilinear grid.
Works with checkpoints saved by train_PCVAE.py.

Usage:
    python visualize_PCVAE.py
    python visualize_PCVAE.py --checkpoint train_PCVAE_results/best_PCVAE.pt \\
                              --n 6 --seed 0 --out_dir figs_PCVAE
    python visualize_PCVAE.py --checkpoint train_PCVAE_results/best_PCVAE.pt \\
                              --sample 42 --out_dir figs_PCVAE
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
from matplotlib.collections import PatchCollection
import torch
from torch import nn
from torch.nn import functional as F

# Ensure relative paths resolve against project root even when launched from
# `utils/`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

try:
    # Works when executed as a package module.
    from ..main_train_pcvae import PCVAE
except ImportError:
    # Works when executed as a script: `python utils/visualize_pcvae.py`.
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from main_train_pcvae import PCVAE

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='train_PCVAE_results/best_PCVAE.pt')
parser.add_argument('--n',          type=int, default=4,  help='test samples to visualise')
parser.add_argument('--sample',     type=int, nargs='+', default=None,
                    help='explicit test sample index/indices (overrides --n random choice)')
parser.add_argument('--seed',       type=int, default=42)
parser.add_argument('--k',          type=int, default=10, help='prior samples averaged per cell')
parser.add_argument('--k_post',     type=int, default=1,
                    help='posterior samples averaged per cell; 1 uses deterministic z=mu')
parser.add_argument('--out_dir',    default='figs_PCVAE')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Grid / model constants (must match train_PCVAE.py)
# ---------------------------------------------------------------------------
NX, NY, NS   = 104, 50, 10
LATENT_SIZE  = 128
COND_SIZE    = 8
HIDDEN       = 2048
FIELD_SIZE   = (2 + 2 * NS) * NX * NY + 1

SPECIES = ['D0', 'D+', 'N0', 'N+', 'N²⁺', 'N³⁺', 'N⁴⁺', 'N⁵⁺', 'N⁶⁺', 'N⁷⁺']

# ---------------------------------------------------------------------------
# Geometry — curvilinear tokamak grid
# ---------------------------------------------------------------------------
crx = np.load(os.path.join('a_dataset', 'geometry', 'crx.npy'))
cry = np.load(os.path.join('a_dataset', 'geometry', 'cry.npy'))

_cells = []
for ix in range(NX):
    for iy in range(NY):
        x = crx[ix, iy, :]
        y = cry[ix, iy, :]
        corners = np.array([[x[0],y[0]], [x[1],y[1]], [x[3],y[3]], [x[2],y[2]]])
        _cells.append(patches.Polygon(corners, closed=True))

xlim = [crx.min(), crx.max()]
ylim = [cry.min(), cry.max()]


def _collection(ax, data_2d, norm, cmap):
    copies = [patches.Polygon(p.get_xy(), closed=True) for p in _cells]
    col = PatchCollection(copies, antialiaseds=False, norm=norm, cmap=cmap, rasterized=True)
    col.set_array(data_2d.flatten())
    ax.add_collection(col)
    return col


def plot_field(ax, data_2d, vmin, vmax, title='', log=True, cmap='viridis', cb=True):
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=7)
    vmin = max(vmin, 1e-40) if log else vmin
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log \
        else mcolors.Normalize(vmin=vmin, vmax=vmax)
    col = _collection(ax, data_2d, norm, cmap)
    if cb:
        plt.colorbar(col, ax=ax, fraction=0.046, pad=0.04)
    return col


def plot_sym(ax, data_2d, vabs, title='', cb=True):
    """Symmetric diverging colormap centred at 0 (for velocities / differences)."""
    vabs = max(vabs, 1e-30)
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=7)
    col = _collection(ax, data_2d, norm, 'RdBu_r')
    if cb:
        plt.colorbar(col, ax=ax, fraction=0.046, pad=0.04)
    return col


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------
_stats = np.load(os.path.join('a_dataset', 'norm_stats_minmax.npz'))
_X_min     = torch.tensor(_stats['X_min'], dtype=torch.float32)
_X_max     = torch.tensor(_stats['X_max'], dtype=torch.float32)
_te_ln_min = float(_stats['te_min'])
_te_ln_max = float(_stats['te_max'])
_ti_ln_min = float(_stats['ti_min'])
_ti_ln_max = float(_stats['ti_max'])
_na_ln_min = torch.tensor(_stats['na_min'], dtype=torch.float32)
_na_ln_max = torch.tensor(_stats['na_max'], dtype=torch.float32)
_ua_min    = torch.tensor(_stats['ua_min'], dtype=torch.float32)
_ua_max    = torch.tensor(_stats['ua_max'], dtype=torch.float32)
_fnixap_train  = np.load(os.path.join('a_dataset', 'train', 'fnixap_tmp.npy'))
_fnixap_ln_min = float(np.log(_fnixap_train.min()))
_fnixap_ln_max = float(np.log(_fnixap_train.max()))

_CHARGE = torch.tensor([0, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)
_M_D    = 2.0  * 1.67262192e-27
_M_N    = 14.0 * 1.67262192e-27
_MASS   = torch.tensor([_M_D]*2 + [_M_N]*8, dtype=torch.float32)


def normalize_X(X):
    return (X - _X_min.to(X.device)) / (_X_max.to(X.device) - _X_min.to(X.device))


def normalize_fields(te, ti, na, ua, fnixap):
    te_n = ((torch.log(te.clamp(min=1e-40)) - _te_ln_min) / (_te_ln_max - _te_ln_min)).view(-1, NX * NY)
    ti_n = ((torch.log(ti.clamp(min=1e-40)) - _ti_ln_min) / (_ti_ln_max - _ti_ln_min)).view(-1, NX * NY)

    na_ln = torch.log(na.clamp(min=1e-40))
    na_n = (na_ln - _na_ln_min.to(na.device)) / (_na_ln_max.to(na.device) - _na_ln_min.to(na.device))
    na_n = na_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)

    ua_n = (ua - _ua_min.to(ua.device)) / (_ua_max.to(ua.device) - _ua_min.to(ua.device))
    ua_n = ua_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)

    fnixap_n = ((torch.log(fnixap.view(-1, 1).clamp(min=1e-40)) - _fnixap_ln_min)
                / (_fnixap_ln_max - _fnixap_ln_min))
    fnixap_n = fnixap_n.expand(-1, NX * NY)

    return torch.cat([te_n, ti_n, na_n, ua_n, fnixap_n], dim=1)


def pack_normalized_fields_to_spatial(x0_flat):
    B = x0_flat.shape[0]
    split = [NX * NY, NX * NY, NS * NX * NY, NS * NX * NY, NX * NY]
    te_n, ti_n, na_n, ua_n, fn_n = torch.split(x0_flat, split, dim=1)
    te_n = te_n.view(B, 1, NX, NY)
    ti_n = ti_n.view(B, 1, NX, NY)
    na_n = na_n.view(B, NS, NX, NY)
    ua_n = ua_n.view(B, NS, NX, NY)
    fn_n = fn_n.view(B, 1, NX, NY)
    return torch.cat([te_n, ti_n, na_n, ua_n, fn_n], dim=1)


def denormalize_fields(x_flat):
    if x_flat.dim() == 4:
        x_flat = x_flat.view(x_flat.shape[0], -1)
    split = [NX*NY, NX*NY, NS*NX*NY, NS*NX*NY, NX*NY]
    te_n, ti_n, na_n, ua_n, fnixap_n = torch.split(x_flat, split, dim=1)
    te = torch.exp(te_n * (_te_ln_max - _te_ln_min) + _te_ln_min).view(-1, NX, NY)
    ti = torch.exp(ti_n * (_ti_ln_max - _ti_ln_min) + _ti_ln_min).view(-1, NX, NY)
    na = torch.exp(na_n.view(-1, NS, NX, NY).permute(0,2,3,1)
                   * (_na_ln_max.to(x_flat.device) - _na_ln_min.to(x_flat.device))
                   + _na_ln_min.to(x_flat.device))
    ua = (ua_n.view(-1, NS, NX, NY).permute(0,2,3,1)
          * (_ua_max.to(x_flat.device) - _ua_min.to(x_flat.device))
          + _ua_min.to(x_flat.device))
    fnixap = torch.exp(fnixap_n * (_fnixap_ln_max - _fnixap_ln_min)
                       + _fnixap_ln_min).mean(dim=1)
    return te, ti, na, ua, fnixap

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=True)

model = PCVAE().to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded '{args.checkpoint}'  epoch={ckpt['epoch']}  val_mse={ckpt['val_mse']:.6f}")

# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
def _load(name, split='test'):
    return torch.tensor(
        np.load(os.path.join('a_dataset', split, f'{name}_tmp.npy')),
        dtype=torch.float32)

X_all      = _load('X')
te_all     = _load('te')
ti_all     = _load('ti')
na_all     = _load('na')
ua_all     = _load('ua')
fnixap_all = _load('fnixap')

if args.sample is not None:
    idx = np.array(args.sample, dtype=int)
    if np.any(idx < 0) or np.any(idx >= len(X_all)):
        raise ValueError(f'--sample indices must be in [0, {len(X_all)-1}]')
    N = len(idx)
else:
    N = min(args.n, len(X_all))
    idx = rng.choice(len(X_all), size=N, replace=False)

print(f'Visualizing sample indices: {idx.tolist()}')

def _batch(t): return t[idx].to(device)

X_b      = _batch(X_all)
te_b     = _batch(te_all)
ti_b     = _batch(ti_all)
na_b     = _batch(na_all)
ua_b     = _batch(ua_all)
fnixap_b = _batch(fnixap_all)

# ---------------------------------------------------------------------------
# Inference — prior mean + posterior (encoder) reconstructions
# ---------------------------------------------------------------------------
K = args.k
K_POST = max(1, args.k_post)
print(f'Averaging K={K} prior samples and K_post={K_POST} posterior samples …')

with torch.no_grad():
    c    = normalize_X(X_b)
    x0_b = pack_normalized_fields_to_spatial(normalize_fields(te_b, ti_b, na_b, ua_b, fnixap_b).clamp(0.0, 1.0))

    acc  = torch.zeros(N, 23, NX, NY, device=device)
    for _ in range(K):
        z   = torch.randn(N, LATENT_SIZE, device=device)
        acc += model.decode(z, c)
    recon_prior = (acc / K).clamp(0.0, 1.0)

    mu, logvar = model.encode(x0_b, c)
    if K_POST == 1:
        recon_post = model.decode(mu, c).clamp(0.0, 1.0)
    else:
        acc_post = torch.zeros(N, 23, NX, NY, device=device)
        for _ in range(K_POST):
            z_post = model.reparameterize(mu, logvar)
            acc_post += model.decode(z_post, c)
        recon_post = (acc_post / K_POST).clamp(0.0, 1.0)

te_pr_t, ti_pr_t, na_pr_t, ua_pr_t, fnixap_pr_t = denormalize_fields(recon_prior)
te_po_t, ti_po_t, na_po_t, ua_po_t, fnixap_po_t = denormalize_fields(recon_post)

# NumPy copies
def np_(t): return t.cpu().numpy()

te_gt   = np_(te_b)
ti_gt   = np_(ti_b)
na_gt   = np_(na_b)      # (N, NX, NY, NS)
ua_gt   = np_(ua_b)      # (N, NX, NY, NS)
fn_gt   = np_(fnixap_b)

te_pr   = np_(te_pr_t);    te_po  = np_(te_po_t)
ti_pr   = np_(ti_pr_t);    ti_po  = np_(ti_po_t)
na_pr   = np_(na_pr_t);    na_po  = np_(na_po_t)
ua_pr   = np_(ua_pr_t);    ua_po  = np_(ua_po_t)
fn_pr   = np_(fnixap_pr_t); fn_po = np_(fnixap_po_t)

# ---------------------------------------------------------------------------
# Helper: save & close
# ---------------------------------------------------------------------------
def _save(fig, name):
    path = os.path.join(args.out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def rmse(a, b):  return np.sqrt(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
def mre(a, b):   return np.mean(np.abs(a-b) / np.abs(b).clip(1e-40))

print(f"\n{'Field':<14}  {'RMSE prior':>12}  {'RMSE post':>12}  {'MRE prior':>12}  {'MRE post':>12}")
print('-' * 74)
for name, gt, pr, po in [('Te [eV]', te_gt, te_pr, te_po), ('Ti [eV]', ti_gt, ti_pr, ti_po)]:
    print(f'{name:<14}  {rmse(gt, pr):>12.3f}  {rmse(gt, po):>12.3f}  {mre(gt, pr):>12.4f}  {mre(gt, po):>12.4f}')
for s in range(NS):
    print(f'na[{SPECIES[s]}]'.ljust(14) +
          f'  {rmse(na_gt[...,s], na_pr[...,s]):>12.3e}'
          f'  {rmse(na_gt[...,s], na_po[...,s]):>12.3e}'
          f'  {mre(na_gt[...,s], na_pr[...,s]):>12.4f}'
          f'  {mre(na_gt[...,s], na_po[...,s]):>12.4f}')
for s in [1, 3]:
    print(f'ua[{SPECIES[s]}]'.ljust(14) +
          f'  {rmse(ua_gt[...,s], ua_pr[...,s]):>12.3e}'
          f'  {rmse(ua_gt[...,s], ua_po[...,s]):>12.3e}'
          f'  {mre(ua_gt[...,s], ua_pr[...,s]):>12.4f}'
          f'  {mre(ua_gt[...,s], ua_po[...,s]):>12.4f}')
print()

# ---------------------------------------------------------------------------
# Fig 1 — Te and Ti  (GT | Prior | Posterior | rel.error prior | rel.error posterior)
# ---------------------------------------------------------------------------
print('Plotting Fig 1: Te / Ti …')
fig, axes = plt.subplots(N, 10, figsize=(36, 3.8*N))
if N == 1: axes = axes[np.newaxis, :]
fig.suptitle(f'Te & Ti  — GT vs prior mean (K={K}) and encoder posterior', fontsize=11)
for i in range(N):
    for j, (gt, pr, po, lbl) in enumerate([(te_gt[i], te_pr[i], te_po[i], '$T_e$'),
                                            (ti_gt[i], ti_pr[i], ti_po[i], '$T_i$')]):
        vmin = max(min(gt.min(), pr.min(), po.min()), 0.1)
        vmax = max(gt.max(), pr.max(), po.max())
        rel_pr = np.abs(pr - gt) / gt.clip(1e-40)
        rel_po = np.abs(po - gt) / gt.clip(1e-40)
        rel_vmax = max(rel_pr.max(), rel_po.max(), 1e-3)
        base = j * 5
        plot_field(axes[i, base + 0], gt, vmin, vmax, title=f'GT {lbl} [eV]' if i == 0 else '')
        plot_field(axes[i, base + 1], pr, vmin, vmax, title=f'Prior {lbl}' if i == 0 else '')
        plot_field(axes[i, base + 2], po, vmin, vmax, title=f'Posterior {lbl}' if i == 0 else '')
        plot_field(axes[i, base + 3], rel_pr.clip(1e-4), 1e-4, rel_vmax,
                   title='Rel err prior' if i == 0 else '', cmap='hot_r')
        plot_field(axes[i, base + 4], rel_po.clip(1e-4), 1e-4, rel_vmax,
                   title='Rel err posterior' if i == 0 else '', cmap='hot_r')
    axes[i, 0].set_ylabel(f'#{idx[i]}', fontsize=8)
fig.tight_layout()
_save(fig, 'fig1_Te_Ti.png')

# ---------------------------------------------------------------------------
# Fig 2 — Total density  n_tot = Σ_s n_s  and electron density n_e = Σ Z_s n_s
#         (GT | Prior | Posterior | rel.error prior | rel.error posterior)
# ---------------------------------------------------------------------------
print('Plotting Fig 2: n_tot / n_e …')
n_tot_gt = na_gt.sum(-1)
n_tot_pr = na_pr.sum(-1)
n_tot_po = na_po.sum(-1)
Z = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
n_e_gt   = (na_gt * Z).sum(-1)
n_e_pr   = (na_pr * Z).sum(-1)
n_e_po   = (na_po * Z).sum(-1)

fig, axes = plt.subplots(N, 10, figsize=(36, 3.8*N))
if N == 1: axes = axes[np.newaxis, :]
fig.suptitle('Total particle density & electron density — GT vs prior and encoder posterior', fontsize=11)
for i in range(N):
    for j, (gt, pr, po, lbl) in enumerate([(n_tot_gt[i], n_tot_pr[i], n_tot_po[i], r'$n_{tot}$'),
                                            (n_e_gt[i],   n_e_pr[i],   n_e_po[i],   r'$n_e$')]):
        vmin = max(min(gt.min(), pr.min(), po.min()), 1e10)
        vmax = max(gt.max(), pr.max(), po.max())
        rel_pr = np.abs(pr - gt) / gt.clip(1e-40)
        rel_po = np.abs(po - gt) / gt.clip(1e-40)
        rel_vmax = max(rel_pr.max(), rel_po.max(), 1e-3)
        base = j * 5
        plot_field(axes[i, base + 0], gt, vmin, vmax, title=f'GT {lbl} [m⁻³]' if i == 0 else '')
        plot_field(axes[i, base + 1], pr, vmin, vmax, title=f'Prior {lbl}' if i == 0 else '')
        plot_field(axes[i, base + 2], po, vmin, vmax, title=f'Posterior {lbl}' if i == 0 else '')
        plot_field(axes[i, base + 3], rel_pr.clip(1e-4), 1e-4, rel_vmax,
                   title='Rel err prior' if i == 0 else '', cmap='hot_r')
        plot_field(axes[i, base + 4], rel_po.clip(1e-4), 1e-4, rel_vmax,
                   title='Rel err posterior' if i == 0 else '', cmap='hot_r')
    axes[i, 0].set_ylabel(f'#{idx[i]}', fontsize=8)
fig.tight_layout()
_save(fig, 'fig2_n_tot_ne.png')

# ---------------------------------------------------------------------------
# Fig 3 — Species densities per sample  (first 2 samples, all 10 species)
# ---------------------------------------------------------------------------
print('Plotting Fig 3: per-species densities …')
for si in range(min(N, 2)):
    fig, axes = plt.subplots(NS, 2, figsize=(8, 3.5*NS))
    fig.suptitle(f'Species densities $n_s$ — sample #{idx[si]}', fontsize=11)
    for s in range(NS):
        gt_s = na_gt[si, :, :, s]
        pr_s = na_pr[si, :, :, s]
        po_s = na_po[si, :, :, s]
        vmin = max(min(gt_s.min(), pr_s.min(), po_s.min()), 1e10)
        vmax = max(gt_s.max(), pr_s.max(), po_s.max())
        plot_field(axes[s, 0], gt_s.clip(vmin), vmin, vmax,
                   title=f'GT $n_s$ [{SPECIES[s]}]' if s==0 else f'GT [{SPECIES[s]}]')
        plot_field(axes[s, 1], pr_s.clip(vmin), vmin, vmax,
                   title=f'Prior [{SPECIES[s]}]' if s==0 else f'Prior [{SPECIES[s]}]')
    fig.tight_layout()
    _save(fig, f'fig3_density_prior_s{idx[si]}.png')

for si in range(min(N, 2)):
    fig, axes = plt.subplots(NS, 2, figsize=(8, 3.5*NS))
    fig.suptitle(f'Species densities $n_s$ (posterior) — sample #{idx[si]}', fontsize=11)
    for s in range(NS):
        gt_s = na_gt[si, :, :, s]
        po_s = na_po[si, :, :, s]
        vmin = max(min(gt_s.min(), po_s.min()), 1e10)
        vmax = max(gt_s.max(), po_s.max())
        plot_field(axes[s, 0], gt_s.clip(vmin), vmin, vmax,
                   title=f'GT $n_s$ [{SPECIES[s]}]' if s==0 else f'GT [{SPECIES[s]}]')
        plot_field(axes[s, 1], po_s.clip(vmin), vmin, vmax,
                   title=f'Posterior [{SPECIES[s]}]' if s==0 else f'Posterior [{SPECIES[s]}]')
    fig.tight_layout()
    _save(fig, f'fig3_density_posterior_s{idx[si]}.png')

# ---------------------------------------------------------------------------
# Fig 4 — Velocities  (D+ and N+, first 2 samples)
# ---------------------------------------------------------------------------
print('Plotting Fig 4: velocities …')
SHOW_UA = [1, 3]  # D+, N+
for si in range(min(N, 2)):
    fig, axes = plt.subplots(len(SHOW_UA), 5, figsize=(19, 4.0*len(SHOW_UA)))
    if len(SHOW_UA) == 1: axes = axes[np.newaxis, :]
    fig.suptitle(f'Velocity $u_{{||}}$ — sample #{idx[si]} (prior and posterior)', fontsize=11)
    for row, s in enumerate(SHOW_UA):
        gt_s = ua_gt[si, :, :, s]
        pr_s = ua_pr[si, :, :, s]
        po_s = ua_po[si, :, :, s]
        vabs = max(np.abs(gt_s).max(), 1.0)
        plot_sym(axes[row, 0], gt_s,        vabs,  title=f'GT $u_{{||}}$ [{SPECIES[s]}] m/s' if row==0 else f'GT [{SPECIES[s]}]')
        plot_sym(axes[row, 1], pr_s,        vabs,  title=f'Prior [{SPECIES[s]}]' if row==0 else f'Prior [{SPECIES[s]}]')
        plot_sym(axes[row, 2], po_s,        vabs,  title=f'Posterior [{SPECIES[s]}]' if row==0 else f'Posterior [{SPECIES[s]}]')
        aerr_pr = pr_s - gt_s
        aerr_po = po_s - gt_s
        err_abs = max(np.abs(aerr_pr).max(), np.abs(aerr_po).max(), 1.0)
        plot_sym(axes[row, 3], aerr_pr, err_abs, title='Error prior [m/s]' if row==0 else '')
        plot_sym(axes[row, 4], aerr_po, err_abs, title='Error posterior [m/s]' if row==0 else '')
    fig.tight_layout()
    _save(fig, f'fig4_velocity_s{idx[si]}.png')

# ---------------------------------------------------------------------------
# Fig 5 — NS momentum flux divergence  ∂(n_s^norm u_s^norm² + n_s^norm T_i^norm)/∂x
#          summed over species  (first 2 samples)
# ---------------------------------------------------------------------------
print('Plotting Fig 5: NS momentum flux divergence …')

def ns_flux_div(flat_norm):
    """Return sum-over-species x-divergence of NS flux, shape (B, NX-1, NY)."""
    if flat_norm.dim() == 4:
        ti_n = flat_norm[:, 1].unsqueeze(-1)                  # (B, NX, NY, 1)
        na_n = flat_norm[:, 2:2 + NS].permute(0, 2, 3, 1)    # (B, NX, NY, NS)
        ua_n = flat_norm[:, 2 + NS:2 + 2 * NS].permute(0, 2, 3, 1)
    else:
        start_ti = NX * NY
        start_na = start_ti + NX * NY
        start_ua = start_na + NS * NX * NY
        ti_n = flat_norm[:, start_ti:start_na].view(-1, NX, NY, 1)
        na_n = flat_norm[:, start_na:start_ua].view(-1, NS, NX, NY).permute(0, 2, 3, 1)
        ua_n = flat_norm[:, start_ua:start_ua + NS * NX * NY].view(-1, NS, NX, NY).permute(0, 2, 3, 1)
    flux = na_n * ua_n.pow(2) + na_n * ti_n                 # (B, NX, NY, NS)
    div  = flux[:, 1:, :, :] - flux[:, :-1, :, :]           # (B, NX-1, NY, NS)
    return div.sum(-1).cpu().numpy()                         # (B, NX-1, NY)


# We need the normalised flat tensors for GT as well — reconstruct from raw fields
def _normalize_fields_raw(te, ti, na, ua, fnixap):
    from torch import log as tlog
    te_n = ((tlog(te.clamp(1e-40)) - _te_ln_min) / (_te_ln_max - _te_ln_min)
            ).view(-1, NX*NY)
    ti_n = ((tlog(ti.clamp(1e-40)) - _ti_ln_min) / (_ti_ln_max - _ti_ln_min)
            ).view(-1, NX*NY)
    na_ln = tlog(na.clamp(1e-40))
    na_n  = ((na_ln - _na_ln_min.to(na.device)) /
             (_na_ln_max.to(na.device) - _na_ln_min.to(na.device)))
    na_n  = na_n.permute(0, 3, 1, 2).reshape(-1, NS*NX*NY)
    ua_n  = ((ua - _ua_min.to(ua.device)) /
             (_ua_max.to(ua.device) - _ua_min.to(ua.device)))
    ua_n  = ua_n.permute(0, 3, 1, 2).reshape(-1, NS*NX*NY)
    fn_n  = ((tlog(fnixap.view(-1,1).clamp(1e-40)) - _fnixap_ln_min) /
             (_fnixap_ln_max - _fnixap_ln_min))
    return torch.cat([te_n, ti_n, na_n, ua_n, fn_n], dim=1).clamp(0, 1)


with torch.no_grad():
    gt_flat  = _normalize_fields_raw(te_b, ti_b, na_b, ua_b, fnixap_b)
    div_gt   = ns_flux_div(gt_flat)            # (N, NX-1, NY)
    div_pr   = ns_flux_div(recon_prior)
    div_po   = ns_flux_div(recon_post)

# Plot on a simple rectilinear grid (NX-1 × NY pixel image)
for si in range(min(N, 2)):
    vabs = max(np.abs(div_gt[si]).max(), np.abs(div_pr[si]).max(), np.abs(div_po[si]).max(), 1e-30)
    eabs = max(np.abs(div_pr[si] - div_gt[si]).max(), np.abs(div_po[si] - div_gt[si]).max(), 1e-30)
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    fig.suptitle(f'NS momentum flux divergence (norm. space) — sample #{idx[si]} (prior/posterior)', fontsize=11)
    for ax, data, title in [
        (axes[0], div_gt[si], 'GT'),
        (axes[1], div_pr[si], 'Prior mean'),
        (axes[2], div_po[si], 'Posterior'),
        (axes[3], div_pr[si] - div_gt[si], 'Prior − GT'),
        (axes[4], div_po[si] - div_gt[si], 'Posterior − GT'),
    ]:
        if 'GT' in title or 'mean' in title or title == 'Posterior':
            vmin, vmax = -vabs, vabs
        else:
            vmin, vmax = -eabs, eabs
        im = ax.imshow(data.T, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='RdBu_r')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x (radial)')
        ax.set_ylabel('y (poloidal)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, f'fig5_ns_div_s{idx[si]}.png')

# ---------------------------------------------------------------------------
# Fig 6 — fnixap scatter (full test set, prior and posterior)
# ---------------------------------------------------------------------------
print('Plotting Fig 6: fnixap scatter (full test set) …')
BATCH = 128
fnixap_pred_all = np.zeros(len(fnixap_all))
fnixap_post_all = np.zeros(len(fnixap_all))
X_all_dev = X_all.to(device)
te_all_dev = te_all.to(device)
ti_all_dev = ti_all.to(device)
na_all_dev = na_all.to(device)
ua_all_dev = ua_all.to(device)
fn_all_dev = fnixap_all.to(device)

with torch.no_grad():
    for start in range(0, len(fnixap_all), BATCH):
        sl  = slice(start, start + BATCH)
        c_b = normalize_X(X_all_dev[sl])
        acc = torch.zeros(c_b.shape[0], device=device)
        for _ in range(K):
            z = torch.randn(c_b.shape[0], LATENT_SIZE, device=device)
            _, _, _, _, fn = denormalize_fields(model.decode(z, c_b))
            acc += fn
        fnixap_pred_all[sl] = (acc / K).cpu().numpy()

        x0_b = pack_normalized_fields_to_spatial(
            normalize_fields(te_all_dev[sl], ti_all_dev[sl], na_all_dev[sl], ua_all_dev[sl], fn_all_dev[sl]).clamp(0.0, 1.0)
        )
        mu_b, _ = model.encode(x0_b, c_b)
        _, _, _, _, fn_post = denormalize_fields(model.decode(mu_b, c_b).clamp(0.0, 1.0))
        fnixap_post_all[sl] = fn_post.cpu().numpy()

fn_gt_all = fnixap_all.numpy()
mask_pr = (fn_gt_all > 0) & (fnixap_pred_all > 0)
mask_po = (fn_gt_all > 0) & (fnixap_post_all > 0)

gt_pr, pr_m = fn_gt_all[mask_pr], fnixap_pred_all[mask_pr]
gt_po, po_m = fn_gt_all[mask_po], fnixap_post_all[mask_po]

vmin_f = min(gt_pr.min(), pr_m.min(), gt_po.min(), po_m.min())
vmax_f = max(gt_pr.max(), pr_m.max(), gt_po.max(), po_m.max())

log_corr_pr = np.corrcoef(np.log(gt_pr), np.log(pr_m))[0, 1]
log_corr_po = np.corrcoef(np.log(gt_po), np.log(po_m))[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].scatter(gt_pr, pr_m, s=4, alpha=0.4)
axes[0].plot([vmin_f, vmax_f], [vmin_f, vmax_f], 'r--', lw=1, label='ideal')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_xlabel(r'GT $\Gamma_{\mathrm{ix}}$ [atoms/s]')
axes[0].set_ylabel(f'Prior mean (K={K}) [atoms/s]')
axes[0].set_title(r'Integrated D ion flux: prior')
axes[0].text(0.05, 0.95, f'log-corr = {log_corr_pr:.3f}', transform=axes[0].transAxes,
             va='top', fontsize=9)
axes[0].legend()

axes[1].scatter(gt_po, po_m, s=4, alpha=0.4)
axes[1].plot([vmin_f, vmax_f], [vmin_f, vmax_f], 'r--', lw=1, label='ideal')
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_xlabel(r'GT $\Gamma_{\mathrm{ix}}$ [atoms/s]')
axes[1].set_ylabel('Posterior mean [atoms/s]')
axes[1].set_title(r'Integrated D ion flux: posterior')
axes[1].text(0.05, 0.95, f'log-corr = {log_corr_po:.3f}', transform=axes[1].transAxes,
             va='top', fontsize=9)
axes[1].legend()

fig.tight_layout()
_save(fig, 'fig6_fnixap.png')

print(f'\nAll figures saved to: {args.out_dir}/')

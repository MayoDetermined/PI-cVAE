"""
Visualize latent-diffusion + PCVAE outputs vs ground truth on the tokamak curvilinear grid.

Usage:
    python visualize_latent_diffusion_pcvae.py
    python visualize_latent_diffusion_pcvae.py --ldm_checkpoint train_LDM_results/best_latent_diffusion.pt
    python visualize_latent_diffusion_pcvae.py --sample 42 130 642 --k 4 --out_dir figs_LDM
"""

import os
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

import sys

# Ensure all relative paths (used in imported training modules too) resolve
# against the project root even when this script is run from `utils/`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

try:
    # Works when executed as a package module.
    from ..main_train_pcvae import PCVAE
    from ..experimental_train_latent_diffusion_pcvae import LatentDenoiser, DiffusionSchedule, UNetDenoiser
except ImportError:
    # Works when executed as a script: `python utils/visualize_pcvae.py`.
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from main_train_pcvae import PCVAE
    from experimental_train_latent_diffusion_pcvae import LatentDenoiser, DiffusionSchedule, UNetDenoiser

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--ldm_checkpoint', default='train_LDM_results/best_latent_diffusion.pt')
parser.add_argument('--pcvae_checkpoint', default=None,
                    help='optional override; if not set, uses value saved in LDM checkpoint')
parser.add_argument('--n', type=int, default=4, help='test samples to visualize')
parser.add_argument('--sample', type=int, nargs='+', default=None,
                    help='explicit test sample index/indices (overrides random choice)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--k', type=int, default=3,
                    help='diffusion samples averaged per selected case')
parser.add_argument('--out_dir', default='figs_LDM')
parser.add_argument('--diff_steps', type=int, default=None,
                    help='optional override of diffusion steps from checkpoint metadata')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NX, NY, NS = 104, 50, 10
LATENT_SIZE = 128
COND_SIZE = 8

SPECIES = ['D0', 'D+', 'N0', 'N+', 'N2+', 'N3+', 'N4+', 'N5+', 'N6+', 'N7+']


# ---------------------------------------------------------------------------
# Geometry (curvilinear tokamak grid)
# ---------------------------------------------------------------------------
crx = np.load(os.path.join('a_dataset', 'geometry', 'crx.npy'))
cry = np.load(os.path.join('a_dataset', 'geometry', 'cry.npy'))

_cells = []
for ix in range(NX):
    for iy in range(NY):
        x = crx[ix, iy, :]
        y = cry[ix, iy, :]
        corners = np.array([[x[0], y[0]], [x[1], y[1]], [x[3], y[3]], [x[2], y[2]]])
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
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=7)
    vmin = max(vmin, 1e-40) if log else vmin
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log else mcolors.Normalize(vmin=vmin, vmax=vmax)
    col = _collection(ax, data_2d, norm, cmap)
    if cb:
        plt.colorbar(col, ax=ax, fraction=0.046, pad=0.04)
    return col


def plot_sym(ax, data_2d, vabs, title='', cb=True):
    vabs = max(vabs, 1e-30)
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=7)
    col = _collection(ax, data_2d, norm, 'RdBu_r')
    if cb:
        plt.colorbar(col, ax=ax, fraction=0.046, pad=0.04)
    return col


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------
_stats = np.load(os.path.join('a_dataset', 'norm_stats_minmax.npz'))
_X_min = torch.tensor(_stats['X_min'], dtype=torch.float32)
_X_max = torch.tensor(_stats['X_max'], dtype=torch.float32)
_te_ln_min = float(_stats['te_min'])
_te_ln_max = float(_stats['te_max'])
_ti_ln_min = float(_stats['ti_min'])
_ti_ln_max = float(_stats['ti_max'])
_na_ln_min = torch.tensor(_stats['na_min'], dtype=torch.float32)
_na_ln_max = torch.tensor(_stats['na_max'], dtype=torch.float32)
_ua_min = torch.tensor(_stats['ua_min'], dtype=torch.float32)
_ua_max = torch.tensor(_stats['ua_max'], dtype=torch.float32)
_fnixap_train = np.load(os.path.join('a_dataset', 'train', 'fnixap_tmp.npy'))
_fnixap_ln_min = float(np.log(_fnixap_train.min()))
_fnixap_ln_max = float(np.log(_fnixap_train.max()))


def normalize_X(X):
    return (X - _X_min.to(X.device)) / (_X_max.to(X.device) - _X_min.to(X.device))


def denormalize_fields(x_flat):
    if x_flat.dim() == 4:
        x_flat = x_flat.view(x_flat.shape[0], -1)

    split = [NX * NY, NX * NY, NS * NX * NY, NS * NX * NY, NX * NY]
    te_n, ti_n, na_n, ua_n, fnixap_n = torch.split(x_flat, split, dim=1)

    te = torch.exp(te_n * (_te_ln_max - _te_ln_min) + _te_ln_min).view(-1, NX, NY)
    ti = torch.exp(ti_n * (_ti_ln_max - _ti_ln_min) + _ti_ln_min).view(-1, NX, NY)

    na = torch.exp(
        na_n.view(-1, NS, NX, NY).permute(0, 2, 3, 1)
        * (_na_ln_max.to(x_flat.device) - _na_ln_min.to(x_flat.device))
        + _na_ln_min.to(x_flat.device)
    )

    ua = (
        ua_n.view(-1, NS, NX, NY).permute(0, 2, 3, 1)
        * (_ua_max.to(x_flat.device) - _ua_min.to(x_flat.device))
        + _ua_min.to(x_flat.device)
    )

    fnixap = torch.exp(fnixap_n * (_fnixap_ln_max - _fnixap_ln_min) + _fnixap_ln_min).mean(dim=1)
    return te, ti, na, ua, fnixap

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args_ = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


@torch.no_grad()
def sample_latents(denoiser, schedule, c):
    B = c.shape[0]
    z = torch.randn(B, LATENT_SIZE, device=c.device)

    for step in reversed(range(schedule.num_steps)):
        t = torch.full((B,), step, device=c.device, dtype=torch.long)

        beta_t = schedule._extract(schedule.betas, t, z.shape)
        alpha_t = schedule._extract(schedule.alphas, t, z.shape)
        abar_t = schedule._extract(schedule.alpha_bars, t, z.shape)
        post_var_t = schedule._extract(schedule.posterior_var, t, z.shape)

        eps_theta = denoiser(z, t, c)
        mean = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - abar_t)) * eps_theta)

        if step > 0:
            z = mean + torch.sqrt(post_var_t) * torch.randn_like(z)
        else:
            z = mean

    return z


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load(name, split='test'):
    return torch.tensor(np.load(os.path.join('a_dataset', split, f'{name}_tmp.npy')), dtype=torch.float32)


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def rmse(a, b):
    return np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def mre(a, b):
    return np.mean(np.abs(a - b) / np.abs(b).clip(1e-40))


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ldm_ckpt = load_checkpoint(args.ldm_checkpoint, device)

pcvae_ckpt_path = args.pcvae_checkpoint or ldm_ckpt.get('pcvae_checkpoint', 'train_PCVAE_results/best_PCVAE.pt')
pcvae_ckpt = load_checkpoint(pcvae_ckpt_path, device)

vae = PCVAE().to(device)
vae.load_state_dict(pcvae_ckpt['model_state_dict'])
vae.eval()

for p in vae.parameters():
    p.requires_grad = False

denoiser = UNetDenoiser().to(device)
denoiser.load_state_dict(ldm_ckpt['denoiser_state_dict'])
denoiser.eval()

beta_start = float(ldm_ckpt.get('beta_start', 1e-4))
beta_end = float(ldm_ckpt.get('beta_end', 2e-2))
ckpt_steps = int(ldm_ckpt.get('diff_steps', 200))
diff_steps = int(args.diff_steps) if args.diff_steps is not None else ckpt_steps

schedule = DiffusionSchedule(
    num_steps=diff_steps,
    beta_start=beta_start,
    beta_end=beta_end,
    device=device,
)

print(f"Loaded LDM checkpoint: {args.ldm_checkpoint}")
print(f"Loaded PCVAE checkpoint: {pcvae_ckpt_path}")
print(f"Diffusion schedule: steps={diff_steps}, beta_start={beta_start}, beta_end={beta_end}")


# ---------------------------------------------------------------------------
# Load test set + sample indices
# ---------------------------------------------------------------------------
X_all = _load('X')
te_all = _load('te')
ti_all = _load('ti')
na_all = _load('na')
ua_all = _load('ua')
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


def _batch(t):
    return t[idx].to(device)


X_b = _batch(X_all)
te_b = _batch(te_all)
ti_b = _batch(ti_all)
na_b = _batch(na_all)
ua_b = _batch(ua_all)
fnixap_b = _batch(fnixap_all)


# ---------------------------------------------------------------------------
# Inference (average K diffusion samples)
# ---------------------------------------------------------------------------
K = args.k
print(f'Averaging K={K} diffusion samples ...')

with torch.no_grad():
    c = normalize_X(X_b)
    acc = torch.zeros(N, 23, NX, NY, device=device)
    for _ in range(K):
        z = sample_latents(denoiser, schedule, c)
        acc += vae.decode(z, c)
    recon_flat = (acc / K).clamp(0.0, 1.0)

te_r, ti_r, na_r, ua_r, fnixap_r = denormalize_fields(recon_flat)


def np_(t):
    return t.cpu().numpy()


te_gt = np_(te_b)
te_rc = np_(te_r)
ti_gt = np_(ti_b)
ti_rc = np_(ti_r)
na_gt = np_(na_b)
na_rc = np_(na_r)
ua_gt = np_(ua_b)
ua_rc = np_(ua_r)
fn_gt = np_(fnixap_b)
fn_rc = np_(fnixap_r)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print(f"\n{'Field':<14}  {'RMSE':>12}  {'Mean rel err':>14}")
print('-' * 44)
for name, gt, rc in [('Te [eV]', te_gt, te_rc), ('Ti [eV]', ti_gt, ti_rc)]:
    print(f'{name:<14}  {rmse(gt, rc):>12.3f}  {mre(gt, rc):>14.4f}')
for s in range(NS):
    print(
        f"na[{SPECIES[s]}]".ljust(14)
        + f'  {rmse(na_gt[..., s], na_rc[..., s]):>12.3e}'
        + f'  {mre(na_gt[..., s], na_rc[..., s]):>14.4f}'
    )
for s in [1, 3]:
    print(
        f"ua[{SPECIES[s]}]".ljust(14)
        + f'  {rmse(ua_gt[..., s], ua_rc[..., s]):>12.3e}'
        + f'  {mre(ua_gt[..., s], ua_rc[..., s]):>14.4f}'
    )
print()


# ---------------------------------------------------------------------------
# Fig 1: Te/Ti
# ---------------------------------------------------------------------------
print('Plotting Fig 1: Te / Ti ...')
fig, axes = plt.subplots(N, 6, figsize=(22, 3.8 * N))
if N == 1:
    axes = axes[np.newaxis, :]
fig.suptitle(f'Te & Ti - GT vs latent diffusion mean (K={K})', fontsize=11)
for i in range(N):
    for j, (gt, rc, lbl) in enumerate([(te_gt[i], te_rc[i], 'Te'), (ti_gt[i], ti_rc[i], 'Ti')]):
        vmin = max(min(gt.min(), rc.min()), 0.1)
        vmax = max(gt.max(), rc.max())
        rel = np.abs(rc - gt) / gt.clip(1e-40)
        plot_field(axes[i, j * 3 + 0], gt, vmin, vmax, title=f'GT {lbl} [eV]' if i == 0 else '')
        plot_field(axes[i, j * 3 + 1], rc, vmin, vmax, title=f'Recon {lbl}' if i == 0 else '')
        plot_field(axes[i, j * 3 + 2], rel.clip(1e-4), 1e-4, rel.max().clip(1e-3),
                   title='Rel err' if i == 0 else '', cmap='hot_r')
    axes[i, 0].set_ylabel(f'#{idx[i]}', fontsize=8)
fig.tight_layout()
_save(fig, args.out_dir, 'fig1_Te_Ti.png')


# ---------------------------------------------------------------------------
# Fig 2: n_tot / n_e
# ---------------------------------------------------------------------------
print('Plotting Fig 2: n_tot / n_e ...')
n_tot_gt = na_gt.sum(-1)
n_tot_rc = na_rc.sum(-1)
Z = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
n_e_gt = (na_gt * Z).sum(-1)
n_e_rc = (na_rc * Z).sum(-1)

fig, axes = plt.subplots(N, 6, figsize=(22, 3.8 * N))
if N == 1:
    axes = axes[np.newaxis, :]
fig.suptitle(f'Total particle density & electron density - GT vs latent diffusion mean (K={K})', fontsize=11)
for i in range(N):
    for j, (gt, rc, lbl) in enumerate([(n_tot_gt[i], n_tot_rc[i], 'n_tot'), (n_e_gt[i], n_e_rc[i], 'n_e')]):
        vmin = max(min(gt.min(), rc.min()), 1e10)
        vmax = max(gt.max(), rc.max())
        rel = np.abs(rc - gt) / gt.clip(1e-40)
        plot_field(axes[i, j * 3 + 0], gt, vmin, vmax, title=f'GT {lbl} [m^-3]' if i == 0 else '')
        plot_field(axes[i, j * 3 + 1], rc, vmin, vmax, title=f'Recon {lbl}' if i == 0 else '')
        plot_field(axes[i, j * 3 + 2], rel.clip(1e-4), 1e-4, rel.max().clip(1e-3),
                   title='Rel err' if i == 0 else '', cmap='hot_r')
    axes[i, 0].set_ylabel(f'#{idx[i]}', fontsize=8)
fig.tight_layout()
_save(fig, args.out_dir, 'fig2_n_tot_ne.png')


# ---------------------------------------------------------------------------
# Fig 3: per-species densities (first 2 cases)
# ---------------------------------------------------------------------------
print('Plotting Fig 3: per-species densities ...')
for si in range(min(N, 2)):
    fig, axes = plt.subplots(NS, 2, figsize=(8, 3.5 * NS))
    fig.suptitle(f'Species densities n_s - sample #{idx[si]}', fontsize=11)
    for s in range(NS):
        gt_s = na_gt[si, :, :, s]
        rc_s = na_rc[si, :, :, s]
        vmin = max(min(gt_s.min(), rc_s.min()), 1e10)
        vmax = max(gt_s.max(), rc_s.max())
        plot_field(axes[s, 0], gt_s.clip(vmin), vmin, vmax,
                   title=f'GT n_s [{SPECIES[s]}]' if s == 0 else f'GT [{SPECIES[s]}]')
        plot_field(axes[s, 1], rc_s.clip(vmin), vmin, vmax,
                   title=f'Recon [{SPECIES[s]}]' if s == 0 else f'Recon [{SPECIES[s]}]')
    fig.tight_layout()
    _save(fig, args.out_dir, f'fig3_density_s{idx[si]}.png')


# ---------------------------------------------------------------------------
# Fig 4: velocities (D+, N+, first 2 cases)
# ---------------------------------------------------------------------------
print('Plotting Fig 4: velocities ...')
SHOW_UA = [1, 3]
for si in range(min(N, 2)):
    fig, axes = plt.subplots(len(SHOW_UA), 3, figsize=(12, 4.0 * len(SHOW_UA)))
    if len(SHOW_UA) == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f'Velocity u_parallel - sample #{idx[si]}', fontsize=11)
    for row, s in enumerate(SHOW_UA):
        gt_s = ua_gt[si, :, :, s]
        rc_s = ua_rc[si, :, :, s]
        vabs = max(np.abs(gt_s).max(), 1.0)
        plot_sym(axes[row, 0], gt_s, vabs, title=f'GT u_parallel [{SPECIES[s]}] m/s' if row == 0 else f'GT [{SPECIES[s]}]')
        plot_sym(axes[row, 1], rc_s, vabs, title=f'Recon [{SPECIES[s]}]' if row == 0 else f'Recon [{SPECIES[s]}]')
        aerr = rc_s - gt_s
        plot_sym(axes[row, 2], aerr, np.abs(aerr).max().clip(1), title='Error [m/s]' if row == 0 else '')
    fig.tight_layout()
    _save(fig, args.out_dir, f'fig4_velocity_s{idx[si]}.png')


# ---------------------------------------------------------------------------
# Fig 5: normalized NS momentum-flux divergence
# ---------------------------------------------------------------------------
print('Plotting Fig 5: NS momentum flux divergence ...')


def ns_flux_div(flat_norm):
    if flat_norm.dim() == 4:
        ti_n = flat_norm[:, 1].unsqueeze(-1)
        na_n = flat_norm[:, 2:2 + NS].permute(0, 2, 3, 1)
        ua_n = flat_norm[:, 2 + NS:2 + 2 * NS].permute(0, 2, 3, 1)
    else:
        start_ti = NX * NY
        start_na = start_ti + NX * NY
        start_ua = start_na + NS * NX * NY
        ti_n = flat_norm[:, start_ti:start_na].view(-1, NX, NY, 1)
        na_n = flat_norm[:, start_na:start_ua].view(-1, NS, NX, NY).permute(0, 2, 3, 1)
        ua_n = flat_norm[:, start_ua:start_ua + NS * NX * NY].view(-1, NS, NX, NY).permute(0, 2, 3, 1)
    flux = na_n * ua_n.pow(2) + na_n * ti_n
    div = flux[:, 1:, :, :] - flux[:, :-1, :, :]
    return div.sum(-1).cpu().numpy()


def _normalize_fields_raw(te, ti, na, ua, fnixap):
    te_n = ((torch.log(te.clamp(1e-40)) - _te_ln_min) / (_te_ln_max - _te_ln_min)).view(-1, NX * NY)
    ti_n = ((torch.log(ti.clamp(1e-40)) - _ti_ln_min) / (_ti_ln_max - _ti_ln_min)).view(-1, NX * NY)

    na_ln = torch.log(na.clamp(1e-40))
    na_n = ((na_ln - _na_ln_min.to(na.device)) / (_na_ln_max.to(na.device) - _na_ln_min.to(na.device)))
    na_n = na_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)

    ua_n = ((ua - _ua_min.to(ua.device)) / (_ua_max.to(ua.device) - _ua_min.to(ua.device)))
    ua_n = ua_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)

    fn_n = ((torch.log(fnixap.view(-1, 1).clamp(1e-40)) - _fnixap_ln_min) / (_fnixap_ln_max - _fnixap_ln_min))
    return torch.cat([te_n, ti_n, na_n, ua_n, fn_n], dim=1).clamp(0, 1)


with torch.no_grad():
    gt_flat = _normalize_fields_raw(te_b, ti_b, na_b, ua_b, fnixap_b)
    div_gt = ns_flux_div(gt_flat)
    div_rc = ns_flux_div(recon_flat)

for si in range(min(N, 2)):
    vabs = max(np.abs(div_gt[si]).max(), np.abs(div_rc[si]).max(), 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'NS momentum flux divergence (norm. space) - sample #{idx[si]}', fontsize=11)
    for ax, data, title in [
        (axes[0], div_gt[si], 'GT'),
        (axes[1], div_rc[si], 'Recon (LDM mean)'),
        (axes[2], div_rc[si] - div_gt[si], 'Recon - GT'),
    ]:
        im = ax.imshow(data.T, origin='lower', aspect='auto', vmin=-vabs, vmax=vabs, cmap='RdBu_r')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x (radial)')
        ax.set_ylabel('y (poloidal)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, args.out_dir, f'fig5_ns_div_s{idx[si]}.png')


# ---------------------------------------------------------------------------
# Fig 6: fnixap scatter (full test set)
# ---------------------------------------------------------------------------
print('Plotting Fig 6: fnixap scatter (full test set) ...')
BATCH = 128
fnixap_pred_all = np.zeros(len(fnixap_all), dtype=np.float64)
X_all_dev = X_all.to(device)

with torch.no_grad():
    for start in range(0, len(fnixap_all), BATCH):
        sl = slice(start, start + BATCH)
        c_b = normalize_X(X_all_dev[sl])
        acc = torch.zeros(c_b.shape[0], device=device)
        for _ in range(K):
            z = sample_latents(denoiser, schedule, c_b)
            _, _, _, _, fn = denormalize_fields(vae.decode(z, c_b))
            acc += fn
        fnixap_pred_all[sl] = (acc / K).cpu().numpy()

fn_gt_all = fnixap_all.numpy()
mask = (fn_gt_all > 0) & (fnixap_pred_all > 0)
gt_m, pr_m = fn_gt_all[mask], fnixap_pred_all[mask]
vmin_f = min(gt_m.min(), pr_m.min())
vmax_f = max(gt_m.max(), pr_m.max())
log_corr = np.corrcoef(np.log(gt_m), np.log(pr_m))[0, 1]

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(gt_m, pr_m, s=4, alpha=0.4)
ax.plot([vmin_f, vmax_f], [vmin_f, vmax_f], 'r--', lw=1, label='ideal')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('GT Gamma_ix [atoms/s]')
ax.set_ylabel(f'LDM mean (K={K}) [atoms/s]')
ax.set_title('Integrated D ion flux Gamma_ix')
ax.text(0.05, 0.95, f'log-corr = {log_corr:.3f}', transform=ax.transAxes, va='top', fontsize=9)
ax.legend()
fig.tight_layout()
_save(fig, args.out_dir, 'fig6_fnixap.png')

print(f'\nAll figures saved to: {args.out_dir}/')

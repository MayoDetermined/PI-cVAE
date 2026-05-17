"""
Compare PCVAE Gaussian prior sampling vs latent diffusion sampling.

Outputs:
- Printed metric table (RMSE + mean relative error) for both methods.
- compare_metrics.csv with per-field metrics.
- compare_fig1_Te_Ti.png with GT | PCVAE prior | LDM (and relative errors).
- compare_fig2_n_tot_ne.png with GT | PCVAE prior | LDM (and relative errors).
- compare_fig3_fnixap_scatter.png on full test set.

Usage:
    python compare_pcvae_vs_ldm.py
    python compare_pcvae_vs_ldm.py --sample 42 130 642 --k_prior 10 --k_ldm 4
"""

import os
import csv
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

import sys

# Ensure all relative paths (including inside imported training modules)
# resolve against the project root when launched from `utils/`.
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
parser.add_argument('--pcvae_checkpoint', default='train_PCVAE_results/best_PCVAE.pt')
parser.add_argument('--ldm_checkpoint', default='train_LDM_results/best_latent_diffusion.pt')
parser.add_argument('--n', type=int, default=4, help='number of test samples (ignored when --sample is used)')
parser.add_argument('--sample', type=int, nargs='+', default=None,
                    help='explicit test sample index/indices')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--k_prior', type=int, default=10, help='MC samples for PCVAE prior mean')
parser.add_argument('--k_ldm', type=int, default=3, help='MC samples for latent diffusion mean')
parser.add_argument('--diff_steps', type=int, default=None,
                    help='override diffusion steps from LDM checkpoint')
parser.add_argument('--out_dir', default='figs_compare')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NX, NY, NS = 104, 50, 10
LATENT_SIZE = 128
COND_SIZE = 8
SPECIES = ['D0', 'D+', 'N0', 'N+', 'N2+', 'N3+', 'N4+', 'N5+', 'N6+', 'N7+']


# ---------------------------------------------------------------------------
# Geometry
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


def denormalize_fields(x_flat_or_spatial):
    if x_flat_or_spatial.dim() == 4:
        x_flat = x_flat_or_spatial.view(x_flat_or_spatial.shape[0], -1)
    else:
        x_flat = x_flat_or_spatial

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
# Utility
# ---------------------------------------------------------------------------
def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def rmse(a, b):
    return np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def mre(a, b):
    return np.mean(np.abs(a - b) / np.abs(b).clip(1e-40))


def save_compare_figure(gt_a, prior_a, ldm_a, gt_b, prior_b, ldm_b, idx, out_path):
    n = gt_a.shape[0]
    fig, axes = plt.subplots(n, 10, figsize=(36, 3.7 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle('GT vs PCVAE prior vs LDM (A and B fields)', fontsize=12)
    for i in range(n):
        fields = [
            ('A', gt_a[i], prior_a[i], ldm_a[i]),
            ('B', gt_b[i], prior_b[i], ldm_b[i]),
        ]

        for j, (label, gt, prior, ldm) in enumerate(fields):
            vmin = max(min(gt.min(), prior.min(), ldm.min()), 1e-20)
            vmax = max(gt.max(), prior.max(), ldm.max())

            rel_prior = np.abs(prior - gt) / gt.clip(1e-40)
            rel_ldm = np.abs(ldm - gt) / gt.clip(1e-40)
            rel_vmax = max(rel_prior.max(), rel_ldm.max(), 1e-3)

            base = j * 5
            plot_field(axes[i, base + 0], gt, vmin, vmax, title=f'GT {label}' if i == 0 else '')
            plot_field(axes[i, base + 1], prior, vmin, vmax, title='PCVAE prior' if i == 0 else '')
            plot_field(axes[i, base + 2], ldm, vmin, vmax, title='LDM' if i == 0 else '')
            plot_field(axes[i, base + 3], rel_prior.clip(1e-4), 1e-4, rel_vmax,
                       title='Rel err prior' if i == 0 else '', cmap='hot_r')
            plot_field(axes[i, base + 4], rel_ldm.clip(1e-4), 1e-4, rel_vmax,
                       title='Rel err LDM' if i == 0 else '', cmap='hot_r')

        axes[i, 0].set_ylabel(f'#{idx[i]}', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    pcvae_ckpt = load_checkpoint(args.pcvae_checkpoint, device)
    ldm_ckpt = load_checkpoint(args.ldm_checkpoint, device)

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

    print(f'Loaded PCVAE: {args.pcvae_checkpoint}')
    print(f'Loaded LDM:   {args.ldm_checkpoint}')
    print(f'Sampling: K_prior={args.k_prior}, K_ldm={args.k_ldm}, diff_steps={diff_steps}')

    def _load(name, split='test'):
        return torch.tensor(np.load(os.path.join('a_dataset', split, f'{name}_tmp.npy')), dtype=torch.float32)

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
    else:
        n = min(args.n, len(X_all))
        idx = rng.choice(len(X_all), size=n, replace=False)

    print(f'Selected indices: {idx.tolist()}')

    X_b = X_all[idx].to(device)
    te_b = te_all[idx].to(device)
    ti_b = ti_all[idx].to(device)
    na_b = na_all[idx].to(device)
    ua_b = ua_all[idx].to(device)
    fn_b = fnixap_all[idx].to(device)

    with torch.no_grad():
        c = normalize_X(X_b)

        acc_prior = torch.zeros(len(idx), 23, NX, NY, device=device)
        for _ in range(args.k_prior):
            z = torch.randn(len(idx), LATENT_SIZE, device=device)
            acc_prior += vae.decode(z, c)
        recon_prior = (acc_prior / args.k_prior).clamp(0.0, 1.0)

        acc_ldm = torch.zeros(len(idx), 23, NX, NY, device=device)
        for _ in range(args.k_ldm):
            z = sample_latents(denoiser, schedule, c)
            acc_ldm += vae.decode(z, c)
        recon_ldm = (acc_ldm / args.k_ldm).clamp(0.0, 1.0)

    te_prior, ti_prior, na_prior, ua_prior, fn_prior = denormalize_fields(recon_prior)
    te_ldm, ti_ldm, na_ldm, ua_ldm, fn_ldm = denormalize_fields(recon_ldm)

    def np_(t):
        return t.detach().cpu().numpy()

    te_gt = np_(te_b)
    ti_gt = np_(ti_b)
    na_gt = np_(na_b)
    ua_gt = np_(ua_b)
    fn_gt = np_(fn_b)

    te_pr = np_(te_prior)
    ti_pr = np_(ti_prior)
    na_pr = np_(na_prior)
    ua_pr = np_(ua_prior)
    fn_pr = np_(fn_prior)

    te_lm = np_(te_ldm)
    ti_lm = np_(ti_ldm)
    na_lm = np_(na_ldm)
    ua_lm = np_(ua_ldm)
    fn_lm = np_(fn_ldm)

    n_tot_gt = na_gt.sum(-1)
    n_tot_pr = na_pr.sum(-1)
    n_tot_lm = na_lm.sum(-1)

    Z = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    n_e_gt = (na_gt * Z).sum(-1)
    n_e_pr = (na_pr * Z).sum(-1)
    n_e_lm = (na_lm * Z).sum(-1)

    rows = []

    def add_metric(name, gt, pred_prior, pred_ldm):
        rows.append({
            'field': name,
            'rmse_prior': rmse(gt, pred_prior),
            'rmse_ldm': rmse(gt, pred_ldm),
            'mre_prior': mre(gt, pred_prior),
            'mre_ldm': mre(gt, pred_ldm),
        })

    add_metric('Te_eV', te_gt, te_pr, te_lm)
    add_metric('Ti_eV', ti_gt, ti_pr, ti_lm)
    add_metric('n_tot_m3', n_tot_gt, n_tot_pr, n_tot_lm)
    add_metric('n_e_m3', n_e_gt, n_e_pr, n_e_lm)
    add_metric('fnixap_atoms_s', fn_gt, fn_pr, fn_lm)

    for s in range(NS):
        add_metric(f'na_{SPECIES[s]}', na_gt[..., s], na_pr[..., s], na_lm[..., s])

    for s in [1, 3]:
        add_metric(f'ua_{SPECIES[s]}', ua_gt[..., s], ua_pr[..., s], ua_lm[..., s])

    print(f"\n{'Field':<16} {'RMSE prior':>14} {'RMSE LDM':>14} {'MRE prior':>12} {'MRE LDM':>12}")
    print('-' * 74)
    for r in rows:
        print(f"{r['field']:<16} {r['rmse_prior']:>14.4e} {r['rmse_ldm']:>14.4e} {r['mre_prior']:>12.4f} {r['mre_ldm']:>12.4f}")

    csv_path = os.path.join(args.out_dir, 'compare_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['field', 'rmse_prior', 'rmse_ldm', 'mre_prior', 'mre_ldm'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved: {csv_path}')

    fig1_path = os.path.join(args.out_dir, 'compare_fig1_Te_Ti.png')
    save_compare_figure(te_gt, te_pr, te_lm, ti_gt, ti_pr, ti_lm, idx, fig1_path)
    print(f'Saved: {fig1_path}')

    fig2_path = os.path.join(args.out_dir, 'compare_fig2_n_tot_ne.png')
    save_compare_figure(n_tot_gt, n_tot_pr, n_tot_lm, n_e_gt, n_e_pr, n_e_lm, idx, fig2_path)
    print(f'Saved: {fig2_path}')

    print('Building full-test fnixap scatter...')
    BATCH = 128
    X_all_dev = X_all.to(device)
    fn_prior_all = np.zeros(len(fnixap_all), dtype=np.float64)
    fn_ldm_all = np.zeros(len(fnixap_all), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(fnixap_all), BATCH):
            sl = slice(start, start + BATCH)
            c_b = normalize_X(X_all_dev[sl])

            acc_p = torch.zeros(c_b.shape[0], device=device)
            for _ in range(args.k_prior):
                z = torch.randn(c_b.shape[0], LATENT_SIZE, device=device)
                _, _, _, _, fn = denormalize_fields(vae.decode(z, c_b))
                acc_p += fn
            fn_prior_all[sl] = (acc_p / args.k_prior).cpu().numpy()

            acc_l = torch.zeros(c_b.shape[0], device=device)
            for _ in range(args.k_ldm):
                z = sample_latents(denoiser, schedule, c_b)
                _, _, _, _, fn = denormalize_fields(vae.decode(z, c_b))
                acc_l += fn
            fn_ldm_all[sl] = (acc_l / args.k_ldm).cpu().numpy()

    fn_gt_all = fnixap_all.numpy()
    mask_prior = (fn_gt_all > 0) & (fn_prior_all > 0)
    mask_ldm = (fn_gt_all > 0) & (fn_ldm_all > 0)

    corr_prior = np.corrcoef(np.log(fn_gt_all[mask_prior]), np.log(fn_prior_all[mask_prior]))[0, 1]
    corr_ldm = np.corrcoef(np.log(fn_gt_all[mask_ldm]), np.log(fn_ldm_all[mask_ldm]))[0, 1]

    vmin_f = max(min(fn_gt_all[fn_gt_all > 0].min(), fn_prior_all[mask_prior].min(), fn_ldm_all[mask_ldm].min()), 1e-40)
    vmax_f = max(fn_gt_all.max(), fn_prior_all.max(), fn_ldm_all.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(fn_gt_all[mask_prior], fn_prior_all[mask_prior], s=4, alpha=0.35)
    axes[0].plot([vmin_f, vmax_f], [vmin_f, vmax_f], 'r--', lw=1)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('GT fnixap [atoms/s]')
    axes[0].set_ylabel(f'PCVAE prior mean K={args.k_prior}')
    axes[0].set_title(f'PCVAE prior, log-corr={corr_prior:.3f}')

    axes[1].scatter(fn_gt_all[mask_ldm], fn_ldm_all[mask_ldm], s=4, alpha=0.35)
    axes[1].plot([vmin_f, vmax_f], [vmin_f, vmax_f], 'r--', lw=1)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('GT fnixap [atoms/s]')
    axes[1].set_ylabel(f'LDM mean K={args.k_ldm}')
    axes[1].set_title(f'LDM, log-corr={corr_ldm:.3f}')

    fig.tight_layout()
    fig3_path = os.path.join(args.out_dir, 'compare_fig3_fnixap_scatter.png')
    fig.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fig3_path}')

    print(f'\nDone. Outputs saved in: {args.out_dir}/')

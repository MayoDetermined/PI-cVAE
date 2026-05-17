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
# Parameter-Conditional VAE  (must match train_PCVAE.py)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return x + h


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.res(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, output_padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_padding
        )
        self.res = ResBlock(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.res(x)
        return x


class PCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(nn.Conv2d(23, 32, 3, padding=1), nn.GELU(), ResBlock(32))
        self.enc_down1 = DownsampleBlock(32, 64)
        self.enc_down2 = DownsampleBlock(64, 128)
        self.enc_down3 = DownsampleBlock(128, 256)

        self.skip_shapes = [(32, 104, 50), (64, 52, 25), (128, 26, 12)]
        self.skip_sizes = [32 * 104 * 50, 64 * 52 * 25, 128 * 26 * 12]
        self.skip_embed_sizes = [32, 64, 128]
        total_skip_embed = sum(self.skip_embed_sizes)

        self.bottleneck_fc1 = nn.Linear(256 * 13 * 6 + total_skip_embed + COND_SIZE, 512)
        self.enc_mu = nn.Linear(512, LATENT_SIZE)
        self.enc_logvar = nn.Linear(512, LATENT_SIZE)

        self.dec_fc1 = nn.Linear(LATENT_SIZE + COND_SIZE, 512)
        self.dec_fc2 = nn.Linear(512, 256 * 13 * 6)

        self.skip_dec_fc = nn.ModuleList([
            nn.Linear(LATENT_SIZE + COND_SIZE, s) for s in self.skip_sizes
        ])
        self.skip_scale = nn.Parameter(torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32))

        self.dec_up3 = UpsampleBlock(256, 128)
        self.dec_up2 = UpsampleBlock(256, 64, output_padding=(0, 1))
        self.dec_up1 = UpsampleBlock(128, 32)

        self.dec_final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 23, 3, padding=1),
            nn.Sigmoid()
        )

        self.act = nn.GELU()

    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = self.act(self.dec_fc1(h))
        h = self.dec_fc2(h)
        h = h.view(-1, 256, 13, 6)

        skip_feats = []
        zc = torch.cat([z, c], dim=1)
        for i, fc in enumerate(self.skip_dec_fc):
            s = fc(zc)
            s = s.view(-1, *self.skip_shapes[i])
            skip_feats.append(self.skip_scale[i] * s)

        h = self.dec_up3(h)
        h = torch.cat([h, skip_feats[2]], dim=1)
        h = self.dec_up2(h)
        h = torch.cat([h, skip_feats[1]], dim=1)
        h = self.dec_up1(h)
        h = torch.cat([h, skip_feats[0]], dim=1)
        return self.dec_final(h)

    def encode(self, x, c):
        raise NotImplementedError('visualize_PCVAE only uses decode()')

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z, c), mu, logvar


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
# Inference — average K prior samples per cell
# ---------------------------------------------------------------------------
K = args.k
print(f'Averaging K={K} prior samples …')

with torch.no_grad():
    c    = normalize_X(X_b)
    acc  = torch.zeros(N, 23, NX, NY, device=device)
    for _ in range(K):
        z   = torch.randn(N, LATENT_SIZE, device=device)
        acc += model.decode(z, c)
    recon_flat = (acc / K).clamp(0.0, 1.0)         # (N, 23, NX, NY) clamped to [0,1]

te_r, ti_r, na_r, ua_r, fnixap_r = denormalize_fields(recon_flat)

# NumPy copies
def np_(t): return t.cpu().numpy()

te_gt   = np_(te_b);       te_rc  = np_(te_r)
ti_gt   = np_(ti_b);       ti_rc  = np_(ti_r)
na_gt   = np_(na_b);       na_rc  = np_(na_r)      # (N, NX, NY, NS)
ua_gt   = np_(ua_b);       ua_rc  = np_(ua_r)      # (N, NX, NY, NS)
fn_gt   = np_(fnixap_b);   fn_rc  = np_(fnixap_r)

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

print(f"\n{'Field':<14}  {'RMSE':>12}  {'Mean rel err':>14}")
print('-' * 44)
for name, gt, rc in [('Te [eV]', te_gt, te_rc), ('Ti [eV]', ti_gt, ti_rc)]:
    print(f'{name:<14}  {rmse(gt, rc):>12.3f}  {mre(gt, rc):>14.4f}')
for s in range(NS):
    print(f'na[{SPECIES[s]}]'.ljust(14) +
          f'  {rmse(na_gt[...,s], na_rc[...,s]):>12.3e}'
          f'  {mre(na_gt[...,s], na_rc[...,s]):>14.4f}')
for s in [1, 3]:
    print(f'ua[{SPECIES[s]}]'.ljust(14) +
          f'  {rmse(ua_gt[...,s], ua_rc[...,s]):>12.3e}'
          f'  {mre(ua_gt[...,s], ua_rc[...,s]):>14.4f}')
print()

# ---------------------------------------------------------------------------
# Fig 1 — Te and Ti  (GT | Recon | rel.error)
# ---------------------------------------------------------------------------
print('Plotting Fig 1: Te / Ti …')
fig, axes = plt.subplots(N, 6, figsize=(22, 3.8*N))
if N == 1: axes = axes[np.newaxis, :]
fig.suptitle(f'Te & Ti  — GT vs prior mean (K={K})', fontsize=11)
for i in range(N):
    for j, (gt, rc, lbl) in enumerate([(te_gt[i], te_rc[i], '$T_e$'),
                                        (ti_gt[i], ti_rc[i], '$T_i$')]):
        vmin = max(min(gt.min(), rc.min()), 0.1)
        vmax = max(gt.max(), rc.max())
        rel  = np.abs(rc - gt) / gt.clip(1e-40)
        plot_field(axes[i, j*3+0], gt,        vmin, vmax, title=f'GT {lbl} [eV]' if i==0 else '')
        plot_field(axes[i, j*3+1], rc,        vmin, vmax, title=f'Recon {lbl}' if i==0 else '')
        plot_field(axes[i, j*3+2], rel.clip(1e-4), 1e-4, rel.max().clip(1e-3),
                   title='Rel err' if i==0 else '', cmap='hot_r')
    axes[i, 0].set_ylabel(f'#{idx[i]}', fontsize=8)
fig.tight_layout()
_save(fig, 'fig1_Te_Ti.png')

# ---------------------------------------------------------------------------
# Fig 2 — Total density  n_tot = Σ_s n_s  and electron density n_e = Σ Z_s n_s
# ---------------------------------------------------------------------------
print('Plotting Fig 2: n_tot / n_e …')
n_tot_gt = na_gt.sum(-1)
n_tot_rc = na_rc.sum(-1)
Z = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
n_e_gt   = (na_gt * Z).sum(-1)
n_e_rc   = (na_rc * Z).sum(-1)

fig, axes = plt.subplots(N, 6, figsize=(22, 3.8*N))
if N == 1: axes = axes[np.newaxis, :]
fig.suptitle(f'Total particle density & electron density — GT vs prior mean (K={K})', fontsize=11)
for i in range(N):
    for j, (gt, rc, lbl) in enumerate([(n_tot_gt[i], n_tot_rc[i], r'$n_{tot}$'),
                                        (n_e_gt[i],   n_e_rc[i],   r'$n_e$')]):
        vmin = max(min(gt.min(), rc.min()), 1e10)
        vmax = max(gt.max(), rc.max())
        rel  = np.abs(rc - gt) / gt.clip(1e-40)
        plot_field(axes[i, j*3+0], gt,        vmin, vmax, title=f'GT {lbl} [m⁻³]' if i==0 else '')
        plot_field(axes[i, j*3+1], rc,        vmin, vmax, title=f'Recon {lbl}' if i==0 else '')
        plot_field(axes[i, j*3+2], rel.clip(1e-4), 1e-4, rel.max().clip(1e-3),
                   title='Rel err' if i==0 else '', cmap='hot_r')
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
        rc_s = na_rc[si, :, :, s]
        vmin = max(min(gt_s.min(), rc_s.min()), 1e10)
        vmax = max(gt_s.max(), rc_s.max())
        plot_field(axes[s, 0], gt_s.clip(vmin), vmin, vmax,
                   title=f'GT $n_s$ [{SPECIES[s]}]' if s==0 else f'GT [{SPECIES[s]}]')
        plot_field(axes[s, 1], rc_s.clip(vmin), vmin, vmax,
                   title=f'Recon [{SPECIES[s]}]' if s==0 else f'Recon [{SPECIES[s]}]')
    fig.tight_layout()
    _save(fig, f'fig3_density_s{idx[si]}.png')

# ---------------------------------------------------------------------------
# Fig 4 — Velocities  (D+ and N+, first 2 samples)
# ---------------------------------------------------------------------------
print('Plotting Fig 4: velocities …')
SHOW_UA = [1, 3]  # D+, N+
for si in range(min(N, 2)):
    fig, axes = plt.subplots(len(SHOW_UA), 3, figsize=(12, 4.0*len(SHOW_UA)))
    if len(SHOW_UA) == 1: axes = axes[np.newaxis, :]
    fig.suptitle(f'Velocity $u_{{||}}$ — sample #{idx[si]}', fontsize=11)
    for row, s in enumerate(SHOW_UA):
        gt_s = ua_gt[si, :, :, s]
        rc_s = ua_rc[si, :, :, s]
        vabs = max(np.abs(gt_s).max(), 1.0)
        plot_sym(axes[row, 0], gt_s,        vabs,  title=f'GT $u_{{||}}$ [{SPECIES[s]}] m/s' if row==0 else f'GT [{SPECIES[s]}]')
        plot_sym(axes[row, 1], rc_s,        vabs,  title=f'Recon [{SPECIES[s]}]' if row==0 else f'Recon [{SPECIES[s]}]')
        aerr = rc_s - gt_s
        plot_sym(axes[row, 2], aerr, np.abs(aerr).max().clip(1),
                 title='Error [m/s]' if row==0 else '')
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
    gt_flat = _normalize_fields_raw(te_b, ti_b, na_b, ua_b, fnixap_b)
    div_gt  = ns_flux_div(gt_flat)            # (N, NX-1, NY)
    div_rc  = ns_flux_div(recon_flat)

# Plot on a simple rectilinear grid (NX-1 × NY pixel image)
for si in range(min(N, 2)):
    vabs = max(np.abs(div_gt[si]).max(), np.abs(div_rc[si]).max(), 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'NS momentum flux divergence (norm. space) — sample #{idx[si]}', fontsize=11)
    for ax, data, title in [
        (axes[0], div_gt[si], 'GT'),
        (axes[1], div_rc[si], 'Recon (prior mean)'),
        (axes[2], div_rc[si] - div_gt[si], 'Recon − GT'),
    ]:
        im = ax.imshow(data.T, origin='lower', aspect='auto',
                       vmin=-vabs, vmax=vabs, cmap='RdBu_r')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x (radial)')
        ax.set_ylabel('y (poloidal)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, f'fig5_ns_div_s{idx[si]}.png')

# ---------------------------------------------------------------------------
# Fig 6 — fnixap scatter (full test set)
# ---------------------------------------------------------------------------
print('Plotting Fig 6: fnixap scatter (full test set) …')
BATCH = 128
fnixap_pred_all = np.zeros(len(fnixap_all))
X_all_dev = X_all.to(device)

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

fn_gt_all = fnixap_all.numpy()
mask = (fn_gt_all > 0) & (fnixap_pred_all > 0)
gt_m, pr_m = fn_gt_all[mask], fnixap_pred_all[mask]
vmin_f = min(gt_m.min(), pr_m.min())
vmax_f = max(gt_m.max(), pr_m.max())
log_corr = np.corrcoef(np.log(gt_m), np.log(pr_m))[0, 1]

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(gt_m, pr_m, s=4, alpha=0.4)
ax.plot([vmin_f, vmax_f], [vmin_f, vmax_f], 'r--', lw=1, label='ideal')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'GT $\Gamma_{\mathrm{ix}}$ [atoms/s]')
ax.set_ylabel(f'Prior mean (K={K}) [atoms/s]')
ax.set_title(r'Integrated D ion flux $\Gamma_{\mathrm{ix}}$')
ax.text(0.05, 0.95, f'log-corr = {log_corr:.3f}', transform=ax.transAxes,
        va='top', fontsize=9)
ax.legend()
fig.tight_layout()
_save(fig, 'fig6_fnixap.png')

print(f'\nAll figures saved to: {args.out_dir}/')

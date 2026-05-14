"""
Visualization script for the trained CVAE model.

Usage:
    python visualize.py                      # uses best_model.pt, 4 random test samples
    python visualize.py --checkpoint path.pt --n 6 --seed 0
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
from matplotlib.collections import PatchCollection
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='best_model.pt')
parser.add_argument('--n', type=int, default=4, help='number of test samples to visualize')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
NX, NY, NS = 104, 50, 10
FIELD_SIZE = (2 + 2*NS) * NX * NY + 1
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


def plot_field(ax, data_2d, vmin, vmax, log=True, title='', colorbar=True,
               cmap='viridis', norm=None):
    """Plot a (NX, NY) field on the curvilinear tokamak grid."""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)
    ax.axis('off')

    if norm is None:
        if log:
            vmin = max(vmin, 1e-40)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cell_copies = [patches.Polygon(p.get_xy(), closed=True) for p in _cells]
    collection = PatchCollection(cell_copies, antialiaseds=False,
                                 norm=norm, cmap=cmap, rasterized=True)
    collection.set_array(data_2d.flatten())
    ax.add_collection(collection)
    if colorbar:
        plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
    return collection


def plot_velocity(ax, data_2d, vabs, title='', colorbar=True):
    """Velocity field with symmetric diverging colormap centred at 0."""
    if vabs < 1.0:
        vabs = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    return plot_field(ax, data_2d, -vabs, vabs, log=False,
                      title=title, colorbar=colorbar, cmap='RdBu_r', norm=norm)


# ---------------------------------------------------------------------------
# Normalization (mirrors cvae_sim.py)
# ---------------------------------------------------------------------------
_stats     = np.load(os.path.join('a_dataset', 'norm_stats_minmax.npz'))
_X_min     = torch.tensor(_stats['X_min'], dtype=torch.float32)
_X_max     = torch.tensor(_stats['X_max'], dtype=torch.float32)
_te_ln_min = float(_stats['te_min'])
_te_ln_max = float(_stats['te_max'])
_ti_ln_min = float(_stats['ti_min'])
_ti_ln_max = float(_stats['ti_max'])
_na_ln_min = torch.tensor(_stats['na_min'], dtype=torch.float32)  # (NS,)
_na_ln_max = torch.tensor(_stats['na_max'], dtype=torch.float32)  # (NS,)
_ua_min    = torch.tensor(_stats['ua_min'], dtype=torch.float32)  # (NS,)
_ua_max    = torch.tensor(_stats['ua_max'], dtype=torch.float32)  # (NS,)

_fnixap_train  = np.load(os.path.join('a_dataset', 'train', 'fnixap_tmp.npy'))
_fnixap_ln_min = float(np.log(_fnixap_train.min()))
_fnixap_ln_max = float(np.log(_fnixap_train.max()))


def normalize_X(X):
    return (X - _X_min.to(X.device)) / (_X_max.to(X.device) - _X_min.to(X.device))


def denormalize_fields(flat):
    split = [NX*NY, NX*NY, NS*NX*NY, NS*NX*NY, 1]
    te_n, ti_n, na_n, ua_n, fnixap_n = torch.split(flat, split, dim=1)
    te = torch.exp(te_n * (_te_ln_max - _te_ln_min) + _te_ln_min).view(-1, NX, NY)
    ti = torch.exp(ti_n * (_ti_ln_max - _ti_ln_min) + _ti_ln_min).view(-1, NX, NY)
    na_mn = _na_ln_min.to(flat.device)
    na_mx = _na_ln_max.to(flat.device)
    na = torch.exp(na_n.view(-1, NS, NX, NY).permute(0,2,3,1) * (na_mx - na_mn) + na_mn)
    ua_mn = _ua_min.to(flat.device)
    ua_mx = _ua_max.to(flat.device)
    ua = ua_n.view(-1, NS, NX, NY).permute(0,2,3,1) * (ua_mx - ua_mn) + ua_mn
    fnixap = torch.exp(fnixap_n * (_fnixap_ln_max - _fnixap_ln_min) + _fnixap_ln_min).view(-1)
    return te, ti, na, ua, fnixap


# ---------------------------------------------------------------------------
# Model â€” architecture MUST match cvae_sim.py
# ---------------------------------------------------------------------------
class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super().__init__()
        self.enc_fc1    = nn.Linear(feature_size + class_size, 2048)
        self.enc_fc2    = nn.Linear(2048, 1024)
        self.enc_mu     = nn.Linear(1024, latent_size)
        self.enc_logvar = nn.Linear(1024, latent_size)
        self.dec_fc1    = nn.Linear(latent_size + class_size, 1024)
        self.dec_fc2    = nn.Linear(1024, 2048)
        self.dec_out    = nn.Linear(2048, feature_size)
        self.act        = nn.ELU()

    def decode(self, z, c):
        h = self.act(self.dec_fc1(torch.cat([z, c], dim=1)))
        h = self.act(self.dec_fc2(h))
        return torch.sigmoid(self.dec_out(h))

    def encode(self, x, c):
        h = self.act(self.enc_fc1(torch.cat([x, c], dim=1)))
        h = self.act(self.enc_fc2(h))
        mu     = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(-10, 4)
        return mu, logvar

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, c), mu, logvar


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=True)

latent_size = ckpt['latent_size']
FIELD_SIZE  = ckpt['field_size']
COND_SIZE   = ckpt['cond_size']

model = CVAE(FIELD_SIZE, latent_size, COND_SIZE).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded '{args.checkpoint}'  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}"
      f"  latent={latent_size}")

# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
X_all      = torch.tensor(np.load(os.path.join('a_dataset', 'test', 'X_tmp.npy')),      dtype=torch.float32)
te_all     = torch.tensor(np.load(os.path.join('a_dataset', 'test', 'te_tmp.npy')),     dtype=torch.float32)
ti_all     = torch.tensor(np.load(os.path.join('a_dataset', 'test', 'ti_tmp.npy')),     dtype=torch.float32)
na_all     = torch.tensor(np.load(os.path.join('a_dataset', 'test', 'na_tmp.npy')),     dtype=torch.float32)
ua_all     = torch.tensor(np.load(os.path.join('a_dataset', 'test', 'ua_tmp.npy')),     dtype=torch.float32)
fnixap_all = torch.tensor(np.load(os.path.join('a_dataset', 'test', 'fnixap_tmp.npy')), dtype=torch.float32)

N   = args.n
idx = rng.choice(len(X_all), size=N, replace=False)

X_batch      = X_all[idx].to(device)
te_batch     = te_all[idx].to(device)
ti_batch     = ti_all[idx].to(device)
na_batch     = na_all[idx].to(device)
ua_batch     = ua_all[idx].to(device)
fnixap_batch = fnixap_all[idx].to(device)

# ---------------------------------------------------------------------------
# Inference â€” production-like prior sampling
# ---------------------------------------------------------------------------
N_PRIOR_SAMPLES = 10

with torch.no_grad():
    c = normalize_X(X_batch)

    recon_acc = torch.zeros(N, FIELD_SIZE, device=device)
    for _ in range(N_PRIOR_SAMPLES):
        z = torch.randn(N, latent_size, device=device)
        recon_acc += model.decode(z, c)
    recon_mean = recon_acc / N_PRIOR_SAMPLES
    te_recon, ti_recon, na_recon, ua_recon, fnixap_recon = denormalize_fields(recon_mean)

# Convert to NumPy
te_gt    = te_batch.cpu().numpy()
ti_gt    = ti_batch.cpu().numpy()
na_gt    = na_batch.cpu().numpy()       # (N, NX, NY, NS)
ua_gt    = ua_batch.cpu().numpy()       # (N, NX, NY, NS)
te_recon = te_recon.cpu().numpy()
ti_recon = ti_recon.cpu().numpy()
na_recon = na_recon.cpu().numpy()
ua_recon = ua_recon.cpu().numpy()
fnixap_gt    = fnixap_batch.cpu().numpy()
fnixap_recon = fnixap_recon.cpu().numpy()

# Temperatures: auto-detect eV vs Joules
eV = 1.602e-19
te_gt_eV    = te_gt    if te_gt.max()    > 1e3 else te_gt    / eV
ti_gt_eV    = ti_gt    if ti_gt.max()    > 1e3 else ti_gt    / eV
te_recon_eV = te_recon if te_recon.max() > 1e3 else te_recon / eV
ti_recon_eV = ti_recon if ti_recon.max() > 1e3 else ti_recon / eV

# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def mean_rel_err(a, b):
    return np.mean(np.abs(a - b) / (np.abs(b).clip(min=1e-40)))

print(f"\n{'Field':<12}  {'RMSE':>14}  {'Mean rel err':>14}")
print("-" * 44)
print(f"{'Te [eV]':<12}  {rmse(te_recon_eV, te_gt_eV):>14.3f}  {mean_rel_err(te_recon_eV, te_gt_eV):>14.3f}")
print(f"{'Ti [eV]':<12}  {rmse(ti_recon_eV, ti_gt_eV):>14.3f}  {mean_rel_err(ti_recon_eV, ti_gt_eV):>14.3f}")
SPECIES = ['D0', 'D1', 'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
for s in range(NS):
    print(f"{'na['+SPECIES[s]+']':<12}  {rmse(na_recon[:,:,:,s], na_gt[:,:,:,s]):>14.3e}"
          f"  {mean_rel_err(na_recon[:,:,:,s], na_gt[:,:,:,s]):>14.3f}")
print()

# ---------------------------------------------------------------------------
# Figure 1: GT vs Prior mean â€” Te and Ti
# ---------------------------------------------------------------------------
fig1, axes1 = plt.subplots(N, 4, figsize=(16, 4.2 * N))
if N == 1:
    axes1 = axes1[np.newaxis, :]
fig1.suptitle(f'GT vs. Prior mean (K={N_PRIOR_SAMPLES}, z~N(0,I))', fontsize=13)

for i in range(N):
    vmin_te = max(min(te_gt_eV[i].min(), te_recon_eV[i].min()), 0.1)
    vmax_te = max(te_gt_eV[i].max(), te_recon_eV[i].max())
    vmin_ti = max(min(ti_gt_eV[i].min(), ti_recon_eV[i].min()), 0.1)
    vmax_ti = max(ti_gt_eV[i].max(), ti_recon_eV[i].max())

    plot_field(axes1[i,0], te_gt_eV[i],    vmin_te, vmax_te,
               title='GT $T_e$ [eV]' if i==0 else '')
    plot_field(axes1[i,1], te_recon_eV[i], vmin_te, vmax_te,
               title=f'Prior mean $T_e$ (K={N_PRIOR_SAMPLES}) [eV]' if i==0 else '')
    plot_field(axes1[i,2], ti_gt_eV[i],    vmin_ti, vmax_ti,
               title='GT $T_i$ [eV]' if i==0 else '')
    plot_field(axes1[i,3], ti_recon_eV[i], vmin_ti, vmax_ti,
               title=f'Prior mean $T_i$ (K={N_PRIOR_SAMPLES}) [eV]' if i==0 else '')
    axes1[i,0].set_ylabel(f'#{idx[i]}', fontsize=8)

fig1.tight_layout()
fig1.savefig('viz_reconstruction.png', dpi=150)
print('Saved viz_reconstruction.png')

# ---------------------------------------------------------------------------
# Figure 2: Relative error â€” Te and Ti
# ---------------------------------------------------------------------------
rel_err_te = np.abs(te_recon - te_gt) / te_gt.clip(min=1e-40)
rel_err_ti = np.abs(ti_recon - ti_gt) / ti_gt.clip(min=1e-40)

fig2, axes2 = plt.subplots(N, 2, figsize=(8, 4.2 * N))
if N == 1:
    axes2 = axes2[np.newaxis, :]
fig2.suptitle(f'Relative error |prior mean â’ GT| / GT  (K={N_PRIOR_SAMPLES})', fontsize=12)

for i in range(N):
    plot_field(axes2[i,0], rel_err_te[i].clip(1e-3),
               1e-3, rel_err_te.max().clip(min=1e-3),
               title='Rel. error $T_e$' if i==0 else '')
    plot_field(axes2[i,1], rel_err_ti[i].clip(1e-3),
               1e-3, rel_err_ti.max().clip(min=1e-3),
               title='Rel. error $T_i$' if i==0 else '')
    axes2[i,0].set_ylabel(f'#{idx[i]}', fontsize=8)

fig2.tight_layout()
fig2.savefig('viz_errors.png', dpi=150)
print('Saved viz_errors.png')

# ---------------------------------------------------------------------------
# Figure 3: Density na â€” GT vs Recon (first 2 samples, species D0 D1 N0 N1)
# ---------------------------------------------------------------------------
SHOW_SPECIES = [0, 1, 2, 3]

for sample_i in range(min(N, 2)):
    fig3, axes3 = plt.subplots(len(SHOW_SPECIES), 2,
                               figsize=(8, 4.2 * len(SHOW_SPECIES)))
    fig3.suptitle(f'Density $n_a$ â€” GT vs Prior mean  (sample #{idx[sample_i]})', fontsize=12)
    for row, s in enumerate(SHOW_SPECIES):
        gt_s  = na_gt[sample_i, :, :, s]
        rec_s = na_recon[sample_i, :, :, s]
        vmin_s = max(min(gt_s.min(), rec_s.min()), 1e10)
        vmax_s = max(gt_s.max(), rec_s.max())
        plot_field(axes3[row, 0], gt_s.clip(vmin_s),  vmin_s, vmax_s,
                   title=f'GT $n_a$ [{SPECIES[s]}] m$^{{-3}}$' if row==0 else f'GT [{SPECIES[s]}]')
        plot_field(axes3[row, 1], rec_s.clip(vmin_s), vmin_s, vmax_s,
                   title=f'Recon $n_a$ [{SPECIES[s]}] m$^{{-3}}$' if row==0 else f'Recon [{SPECIES[s]}]')
    fig3.tight_layout()
    fname = f'viz_density_sample{idx[sample_i]}.png'
    fig3.savefig(fname, dpi=150)
    print(f'Saved {fname}')

# ---------------------------------------------------------------------------
# Figure 4: Velocity ua â€” GT | Recon | |Error|  (RdBu_r diverging colormap)
# ---------------------------------------------------------------------------
SHOW_UA = [1, 3]   # D1 (ion), N1

for sample_i in range(min(N, 2)):
    fig4, axes4 = plt.subplots(len(SHOW_UA), 3,
                               figsize=(12, 4.2 * len(SHOW_UA)))
    if len(SHOW_UA) == 1:
        axes4 = axes4[np.newaxis, :]
    fig4.suptitle(f'Parallel velocity $u_{{||}}$ â€” GT vs Recon  (sample #{idx[sample_i]})',
                  fontsize=12)

    for row, s in enumerate(SHOW_UA):
        gt_s  = ua_gt[sample_i, :, :, s]
        rec_s = ua_recon[sample_i, :, :, s]
        vabs  = max(np.abs(gt_s).max(), 1.0)
        plot_velocity(axes4[row, 0], gt_s,  vabs,
                      title=f'GT $u_{{||}}$ [{SPECIES[s]}] m/s' if row==0 else f'GT [{SPECIES[s]}]')
        plot_velocity(axes4[row, 1], rec_s, vabs,
                      title=f'Recon [{SPECIES[s]}] m/s' if row==0 else f'Recon [{SPECIES[s]}]')
        aerr = np.abs(rec_s - gt_s)
        plot_field(axes4[row, 2], aerr.clip(1), 1.0, aerr.max().clip(min=2),
                   title='|Error| m/s' if row==0 else '', cmap='hot_r')

    fig4.tight_layout()
    fname = f'viz_velocity_sample{idx[sample_i]}.png'
    fig4.savefig(fname, dpi=150)
    print(f'Saved {fname}')

# ---------------------------------------------------------------------------
# Figure 5: fnixap scatter â€” GT vs prior mean (full test set, log-log)
# ---------------------------------------------------------------------------
fnixap_pred_all = np.zeros(len(fnixap_all))
X_all_dev = X_all.to(device)
BATCH = 128
with torch.no_grad():
    for start in range(0, len(fnixap_all), BATCH):
        sl  = slice(start, start + BATCH)
        c_b = normalize_X(X_all_dev[sl])
        acc = torch.zeros(c_b.shape[0], device=device)
        for _ in range(N_PRIOR_SAMPLES):
            z = torch.randn(c_b.shape[0], latent_size, device=device)
            _, _, _, _, fn = denormalize_fields(model.decode(z, c_b))
            acc += fn
        fnixap_pred_all[sl] = (acc / N_PRIOR_SAMPLES).cpu().numpy()

all_fnixap_gt = fnixap_all.numpy()
mask  = (all_fnixap_gt > 0) & (fnixap_pred_all > 0)
gt_m, pr_m = all_fnixap_gt[mask], fnixap_pred_all[mask]
vmin_f = min(gt_m.min(), pr_m.min())
vmax_f = max(gt_m.max(), pr_m.max())

fig5, ax5 = plt.subplots(figsize=(5, 5))
ax5.scatter(gt_m, pr_m, s=4, alpha=0.4)
ax5.plot([vmin_f, vmax_f], [vmin_f, vmax_f], 'r--', lw=1, label='ideal')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlabel('GT $\\Gamma_{\\mathrm{ix}}$ [atoms/s]')
ax5.set_ylabel(f'Prior mean (K={N_PRIOR_SAMPLES}) [atoms/s]')
ax5.set_title('Integrated D ion flux $\\Gamma_{\\mathrm{ix}}$')
log_corr = np.corrcoef(np.log(gt_m), np.log(pr_m))[0, 1]
ax5.text(0.05, 0.95, f'log-corr = {log_corr:.3f}', transform=ax5.transAxes,
         va='top', fontsize=9)
ax5.legend()
fig5.tight_layout()
fig5.savefig('viz_fnixap.png', dpi=150)
print('Saved viz_fnixap.png')

plt.show()

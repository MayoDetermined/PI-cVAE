"""
CNN-based Parameter-Conditional VAE with:
  - Residual blocks + U-Net skip connections (throught latent to prevent collaps!!)
  - GELU activation
  - Physics losses: energy, continuity, navier_stokes, geometry balance
  - Parameters concatenated in bottleneck + decoder
  - Warmup for both KLD and physics losses
"""

import os, math, argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',      type=int, default=150)
parser.add_argument('--results_dir', type=str, default='train_PCVAE_results')
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)
CKPT_PATH = os.path.join(args.results_dir, 'best_PCVAE.pt')

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader_kwargs = {'num_workers': 0, 'pin_memory': torch.cuda.is_available()}
print(f'Device: {device}')

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE  = 64
LR          = 1e-3
LATENT_SIZE = 128
#HIDDEN      = 2048
BETA_KLD    = 0.01
FREE_BITS   = 0.5
BETA_WARMUP = 1        # KL capacity warm-up epochs
KLD_CAPACITY_MAX = 3.0
PHYS_WARMUP = 30        # physics losses warm-up epochs
N_EVAL      = 5         # generated samples per test point

NX, NY, NS  = 104, 50, 10
FIELD_SIZE  = (2 + 2*NS) * NX * NY + 1
COND_SIZE   = 8

# Auxiliary loss lambdas
LAMBDA_RECON        = 1.0
LAMBDA_ENERGY       = 1e-3
LAMBDA_CONTINUITY   = 1e-3
LAMBDA_NAVIER_STOKES = 1e-3
LAMBDA_GEOMETRY_BALANCE = 5e-4

# ---------------------------------------------------------------------------
# Normalization stats
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

_crx = torch.tensor(np.load(os.path.join('a_dataset', 'geometry', 'crx.npy')), dtype=torch.float32)
_cry = torch.tensor(np.load(os.path.join('a_dataset', 'geometry', 'cry.npy')), dtype=torch.float32)


def _cell_area_and_metrics(crx, cry):
    """Return cell areas and center-to-center spacings for the structured tokamak grid."""
    x0, x1, x2, x3 = crx[..., 0], crx[..., 1], crx[..., 2], crx[..., 3]
    y0, y1, y2, y3 = cry[..., 0], cry[..., 1], cry[..., 2], cry[..., 3]
    area = 0.5 * torch.abs(
        x0 * y1 + x1 * y3 + x3 * y2 + x2 * y0
        - y0 * x1 - y1 * x3 - y3 * x2 - y2 * x0
    )
    cx = crx.mean(dim=-1)
    cy = cry.mean(dim=-1)
    dx = torch.sqrt((cx[1:, :] - cx[:-1, :]).pow(2) + (cy[1:, :] - cy[:-1, :]).pow(2)).clamp(min=1e-12)
    dy = torch.sqrt((cx[:, 1:] - cx[:, :-1]).pow(2) + (cy[:, 1:] - cy[:, :-1]).pow(2)).clamp(min=1e-12)
    area_x = 0.5 * (area[1:, :] + area[:-1, :]).clamp(min=1e-20)
    area_y = 0.5 * (area[:, 1:] + area[:, :-1]).clamp(min=1e-20)
    return area.clamp(min=1e-20), dx, dy, area_x, area_y


_CELL_AREA, _DX_X, _DY_Y, _AREA_X, _AREA_Y = _cell_area_and_metrics(_crx, _cry)

_M_D  = 2.0  * 1.67262192e-27
_M_N  = 14.0 * 1.67262192e-27
_eV   = 1.602176634e-19
_MASS = torch.tensor([_M_D, _M_D, _M_N, _M_N, _M_N, _M_N, _M_N, _M_N, _M_N, _M_N],
                     dtype=torch.float32)
_CHARGE = torch.tensor([0, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)


def normalize_X(X):
    return (X - _X_min.to(X.device)) / (_X_max.to(X.device) - _X_min.to(X.device))


def _log_minmax(field, ln_min, ln_max):
    return (torch.log(field.clamp(min=1e-40)) - ln_min) / (ln_max - ln_min)


def _weighted_mse(pred, target, weight):
    """Mean squared error weighted by a broadcastable geometry tensor."""
    w = weight.to(pred.device)
    if w.dim() == 2 and pred.dim() == 4:
        w = w[None, :, :, None]
    else:
        while w.dim() < pred.dim():
            w = w.unsqueeze(0)
    w = w.expand_as(pred)
    return ((pred - target).pow(2) * w).mean() / w.mean().clamp(min=1e-40)

# This helps to deal whith freaky gradients 
def _metric_residual_2d(pred_r, pred_t, dx, dy, area_x, area_y):
    """Compare physical x/y gradients of a scalar or per-species field.

    The residual is weighted by the local cell area and normalized by a target-based
    scale so that large-magnitude channels do not dominate the loss.
    """
    if pred_r.dim() == 3:
        dx = dx.to(pred_r.device)[None, :, :]
        dy = dy.to(pred_r.device)[None, :, :]

        area_x = area_x.to(pred_r.device)
        area_y = area_y.to(pred_r.device)
    else:
        dx = dx.to(pred_r.device)[None, :, :, None]
        dy = dy.to(pred_r.device)[None, :, :, None]

        area_x = area_x.to(pred_r.device)[None, :, :, None]
        area_y = area_y.to(pred_r.device)[None, :, :, None]

    grad_r_x = (pred_r[:, 1:, ...] - pred_r[:, :-1, ...]) / dx
    grad_t_x = (pred_t[:, 1:, ...] - pred_t[:, :-1, ...]) / dx
    grad_r_y = (pred_r[:, :, 1:, ...] - pred_r[:, :, :-1, ...]) / dy
    grad_t_y = (pred_t[:, :, 1:, ...] - pred_t[:, :, :-1, ...]) / dy

    scale_x = grad_t_x.detach().abs().mean(dim=[0, 1, 2], keepdim=True).clamp(min=1e-40)
    scale_y = grad_t_y.detach().abs().mean(dim=[0, 1, 2], keepdim=True).clamp(min=1e-40)
    return 0.5 * (
        _weighted_mse(grad_r_x / scale_x, grad_t_x / scale_x, area_x)
        + _weighted_mse(grad_r_y / scale_y, grad_t_y / scale_y, area_y)
    )


def normalize_fields(te, ti, na, ua, fnixap):
    te_n = _log_minmax(te, _te_ln_min, _te_ln_max).view(-1, NX * NY)
    ti_n = _log_minmax(ti, _ti_ln_min, _ti_ln_max).view(-1, NX * NY)

    na_ln = torch.log(na.clamp(min=1e-40))
    na_n  = (na_ln - _na_ln_min.to(na.device)) / (_na_ln_max.to(na.device) - _na_ln_min.to(na.device))
    na_n  = na_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)

    ua_n  = (ua - _ua_min.to(ua.device)) / (_ua_max.to(ua.device) - _ua_min.to(ua.device))
    ua_n  = ua_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)

    fnixap_n = _log_minmax(fnixap.view(-1, 1), _fnixap_ln_min, _fnixap_ln_max)
    fnixap_n = fnixap_n.expand(-1, NX * NY)  # Expand to spatial shape
    return torch.cat([te_n, ti_n, na_n, ua_n, fnixap_n], dim=1)


def denormalize_fields(x_flat_or_spatial):
    """Accept both flat (B, FIELD_SIZE) and spatial (B, C, H, W) formats."""
    if x_flat_or_spatial.dim() == 4:
        # (B, C, H, W) format - reshape to flat first
        B = x_flat_or_spatial.shape[0]
        x_flat = x_flat_or_spatial.view(B, -1)
    else:
        # (B, FIELD_SIZE) format
        x_flat = x_flat_or_spatial
        B = x_flat.shape[0]
    
    split = [NX*NY, NX*NY, NS*NX*NY, NS*NX*NY, NX*NY]  # fnixap as spatial channel
    te_n, ti_n, na_n, ua_n, fnixap_n = torch.split(x_flat, split, dim=1)

    te = torch.exp(te_n * (_te_ln_max - _te_ln_min) + _te_ln_min).view(-1, NX, NY)
    ti = torch.exp(ti_n * (_ti_ln_max - _ti_ln_min) + _ti_ln_min).view(-1, NX, NY)

    na_n = na_n.view(-1, NS, NX, NY).permute(0, 2, 3, 1)
    na   = torch.exp(na_n * (_na_ln_max.to(x_flat.device) - _na_ln_min.to(x_flat.device))
                     + _na_ln_min.to(x_flat.device))

    ua_n = ua_n.view(-1, NS, NX, NY).permute(0, 2, 3, 1)
    ua   = ua_n * (_ua_max.to(x_flat.device) - _ua_min.to(x_flat.device)) + _ua_min.to(x_flat.device)

    fnixap = torch.exp(fnixap_n * (_fnixap_ln_max - _fnixap_ln_min) + _fnixap_ln_min).view(-1)
    return te, ti, na, ua, fnixap


def prepare_batch(batch):
    X, te, ti, na, ua, fnixap = [t.to(device) for t in batch]

    c  = normalize_X(X)
    # Normalize fields and reshape to (B, C, H, W)
    x0_flat = normalize_fields(te, ti, na, ua, fnixap).clamp(0.0, 1.0)
    # Unflatten: split and reshape to (B, C, NX, NY)
    B = x0_flat.shape[0]
    split = [NX*NY, NX*NY, NS*NX*NY, NS*NX*NY, NX*NY]  # fnixap expanded to spatial

    te_n, ti_n, na_n, ua_n, fn_n = torch.split(x0_flat, split, dim=1)

    te_n = te_n.view(B, 1, NX, NY)
    ti_n = ti_n.view(B, 1, NX, NY)
    na_n = na_n.view(B, NS, NX, NY)
    ua_n = ua_n.view(B, NS, NX, NY)
    fn_n = fn_n.view(B, 1, NX, NY)  # Spatial channel

    x0 = torch.cat([te_n, ti_n, na_n, ua_n, fn_n], dim=1)  # (B, 2+NS+NS+1, NX, NY) = (B, 23, NX, NY)

    return x0, c


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SimDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        base = os.path.join('a_dataset', split)

        self.X      = torch.tensor(np.load(os.path.join(base, 'X_tmp.npy')),      dtype=torch.float32)
        self.te     = torch.tensor(np.load(os.path.join(base, 'te_tmp.npy')),     dtype=torch.float32)
        self.ti     = torch.tensor(np.load(os.path.join(base, 'ti_tmp.npy')),     dtype=torch.float32)
        self.na     = torch.tensor(np.load(os.path.join(base, 'na_tmp.npy')),     dtype=torch.float32)
        self.ua     = torch.tensor(np.load(os.path.join(base, 'ua_tmp.npy')),     dtype=torch.float32)
        self.fnixap = torch.tensor(np.load(os.path.join(base, 'fnixap_tmp.npy')), dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.te[idx], self.ti[idx], self.na[idx], self.ua[idx], self.fnixap[idx]


train_loader = torch.utils.data.DataLoader(
    SimDataset('train'), batch_size=BATCH_SIZE, shuffle=True,  **loader_kwargs)
test_loader  = torch.utils.data.DataLoader(
    SimDataset('test'),  batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

# ---------------------------------------------------------------------------
# Physics losses
# ---------------------------------------------------------------------------
def physics_loss_energy(recon, target):
    """Geometry-aware total energy-density residual.

    This uses electron density for the electron thermal term, total ion density for
    the ion thermal term, and the species-resolved kinetic energy summed over all
    species. The residual is then matched in both x and y on the tokamak metric.
    """
    te_r, ti_r, na_r, ua_r, _ = denormalize_fields(recon)
    te_t, ti_t, na_t, ua_t, _ = denormalize_fields(target)

    mass   = _MASS.to(recon.device)

    n_e_r = (na_r * _CHARGE.to(recon.device)).sum(dim=-1)
    n_e_t = (na_t * _CHARGE.to(recon.device)).sum(dim=-1)

    n_i_r = na_r.sum(dim=-1)
    n_i_t = na_t.sum(dim=-1)

    kin_r = 0.5 * (na_r * ua_r.pow(2) * mass).sum(dim=-1)
    kin_t = 0.5 * (na_t * ua_t.pow(2) * mass).sum(dim=-1)

    E_r = 1.5 * (n_e_r * te_r + n_i_r * ti_r) * _eV + kin_r
    E_t = 1.5 * (n_e_t * te_t + n_i_t * ti_t) * _eV + kin_t
    return _metric_residual_2d(E_r, E_t, _DX_X, _DY_Y, _AREA_X, _AREA_Y)


def physics_loss_continuity(recon, target):
    """Full 2D residual of the species particle-flux (n_s * u_s).
    
    Enforces particle conservation by matching the particle flux across the domain.
    The particle flux j = n*u represents the flow of species particles, and this loss
    ensures that the reconstructed density and velocity profiles conserve particles
    in both spatial directions on the tokamak mesh.
    """
    _, _, na_r, ua_r, _ = denormalize_fields(recon)
    _, _, na_t, ua_t, _ = denormalize_fields(target)
    j_r = na_r * ua_r
    j_t = na_t * ua_t
    return _metric_residual_2d(j_r, j_t, _DX_X, _DY_Y, _AREA_X, _AREA_Y)


def physics_loss_navier_stokes(recon, target):
    """Full 2D residual of the ion momentum-flux. Derved from NS fluid equation (this is implemented instead of global momentum loss).

    This keeps the existing scalar momentum-flux model but matches it in both
    coordinate directions on the curvilinear mesh.
    """
    _, ti_r, na_r, ua_r, _ = denormalize_fields(recon)
    _, ti_t, na_t, ua_t, _ = denormalize_fields(target)

    mass   = _MASS.to(recon.device)

    mom_flux_r = na_r * ua_r.pow(2) * mass + na_r * ti_r.unsqueeze(-1) * _eV
    mom_flux_t = na_t * ua_t.pow(2) * mass + na_t * ti_t.unsqueeze(-1) * _eV

    return _metric_residual_2d(mom_flux_r, mom_flux_t, _DX_X, _DY_Y, _AREA_X, _AREA_Y)


def geometry_balance_loss(recon, target):
    """Global cell-integrated balance terms weighted by tokamak cell areas.

    This checks whether the reconstruction matches the target in area-integrated
    energy, mass flux and momentum flux, which is a coarse but physically meaningful
    global constraint on the curvilinear mesh.
    """
    # Denormalize reconstructed and target fields to physical units
    # Fields: electron temperature (te), ion temperature (ti), 
    #         density (na), velocity (ua), and one unused field
    te_r, ti_r, na_r, ua_r, _ = denormalize_fields(recon)
    te_t, ti_t, na_t, ua_t, _ = denormalize_fields(target)
    
    # Load physical constants (mass, charge in SI units, eV conversion factor, stuff)
    mass   = _MASS.to(recon.device)
    charge = _CHARGE.to(recon.device)
    
    # Load and reshape cell areas for tokamak mesh
    # Shape: [1, nx, ny, 1] for broadcasting with field tensors [batch, nx, ny, n_species]
    area = _CELL_AREA.to(recon.device)[None, :, :, None]

    # Total energy density (thermal + kinetic) for reconstructed fields
    # E = 1.5 * (n_e * T_e + n_i * T_i) + 0.5 * n_i * v^2

    # First term: 1.5 * thermal energy (3 degrees of freedom in 2D + time-dependent)
    # Second term: kinetic energy per unit volume
    E_r = (1.5 * (charge * na_r * te_r.unsqueeze(-1) * _eV
                + na_r * ti_r.unsqueeze(-1) * _eV)
           + 0.5 * na_r * ua_r.pow(2) * mass)
    
    # Same calculation for target fields (ground truth, GT)
    E_t = (1.5 * (charge * na_t * te_t.unsqueeze(-1) * _eV
                + na_t * ti_t.unsqueeze(-1) * _eV)
           + 0.5 * na_t * ua_t.pow(2) * mass)
    
    # Mass flux (particle flow) per unit area
    # j = n * v (number of particles per unit time per unit area)
    j_r = na_r * ua_r  # reconstructed mass flux
    j_t = na_t * ua_t  # target mass flux

    # Momentum flux (pressure + convective momentum transport)
    # mom = n * v^2 * m + n * T

    # First term: convective momentum (kinetic energy flux)
    # Second term: thermal pressure term
    mom_r = na_r * ua_r.pow(2) * mass + na_r * ti_r.unsqueeze(-1) * _eV
    mom_t = na_t * ua_t.pow(2) * mass + na_t * ti_t.unsqueeze(-1) * _eV

    # Integrate quantities over entire mesh using cell areas as weights
    # sum(dim=(1, 2)) integrates over spatial dimensions (x, y), keeping batch and species dims
    # Result shape: [batch, n_species]
    E_int_r = (E_r * area).sum(dim=(1, 2))      # total energy integrated over domain (recon)
    E_int_t = (E_t * area).sum(dim=(1, 2))      # total energy integrated over domain (target)

    j_int_r = (j_r * area).sum(dim=(1, 2))      # total mass flux integrated (recon)
    j_int_t = (j_t * area).sum(dim=(1, 2))      # total mass flux integrated (target)

    mom_int_r = (mom_r * area).sum(dim=(1, 2))  # total momentum integrated (recon)
    mom_int_t = (mom_t * area).sum(dim=(1, 2))  # total momentum integrated (target)

    # Compute normalization scales to prevent numerical instability
    # detach(): stop gradients to prevent feedback loop in normalization
    
    # abs().mean(dim=0): average absolute value across batch
    # clamp(min=1e-40): prevent division by zero with small epsilon value
    scale_E = E_int_t.detach().abs().mean(dim=0, keepdim=True).clamp(min=1e-40)
    scale_j = j_int_t.detach().abs().mean(dim=0, keepdim=True).clamp(min=1e-40)
    scale_m = mom_int_t.detach().abs().mean(dim=0, keepdim=True).clamp(min=1e-40)

    # Compute normalized MSE losses for each conserved quantity
    # Normalization ensures equal weighting regardless of absolute magnitude differences
    loss_E = F.mse_loss(E_int_r / scale_E, E_int_t / scale_E)      # energy conservation loss
    loss_j = F.mse_loss(j_int_r / scale_j, j_int_t / scale_j)      # mass continuity loss
    loss_m = F.mse_loss(mom_int_r / scale_m, mom_int_t / scale_m)  # momentum conservation loss
    
    # Return average of all three balance losses as a composite constraint
    return (loss_E + loss_j + loss_m) / 3.0


def aux_losses(recon, target):
    e   = physics_loss_energy(recon, target)
    c   = physics_loss_continuity(recon, target)
    ns  = physics_loss_navier_stokes(recon, target)
    gb  = geometry_balance_loss(recon, target)

    total = (LAMBDA_ENERGY * e
           + LAMBDA_CONTINUITY * c
           + LAMBDA_NAVIER_STOKES * ns
           + LAMBDA_GEOMETRY_BALANCE * gb)
    return total, e, c, ns, gb


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Model: CNN-based Parameter-Conditional VAE 
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual block with GELU activation."""
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
    """Downsampling: stride-2 conv + residual block."""
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
    """Upsampling: transposed conv + residual block."""
    def __init__(self, in_ch, out_ch, output_padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_padding)
        self.res = ResBlock(out_ch)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.res(x)
        return x


class PCVAE(nn.Module):
    """CNN-based Parameter-Conditional VAE (weakly-conditional: parameters only in bottleneck + decoder).
    
    Architecture:
    - Encoder: spatial data (x) -> latent distribution (z-independent of condition)
    - Bottleneck: concatenate [latent_z || condition_c]
    - Decoder: [z || c] -> reconstruction
    
    This design allows learning shared latent space across conditions with parameter influence
    limited to the generative (decoder) process.
    """
    def __init__(self):
        super().__init__()
        # Input: (B, 23, 104, 50)
        # Encoder
        self.enc_conv0 = nn.Sequential(nn.Conv2d(23, 32, 3, padding=1), nn.GELU(), ResBlock(32))
        self.enc_down1 = DownsampleBlock(32, 64)      # (B, 64, 52, 25)
        self.enc_down2 = DownsampleBlock(64, 128)     # (B, 128, 26, 12)
        self.enc_down3 = DownsampleBlock(128, 256)    # (B, 256, 13, 6)

        # Bottleneck: use compact pooled skip embeddings instead of full flattened maps
        self.skip_shapes = [(32, 104, 50), (64, 52, 25), (128, 26, 12)]
        self.skip_sizes = [32*104*50, 64*52*25, 128*26*12]
        self.skip_embed_sizes = [32, 64, 128]

        total_skip_embed = sum(self.skip_embed_sizes)

        self.bottleneck_fc1 = nn.Linear(256 * 13 * 6 + total_skip_embed + COND_SIZE, 512)
        self.enc_mu = nn.Linear(512, LATENT_SIZE)
        self.enc_logvar = nn.Linear(512, LATENT_SIZE)

        # Decoder: latent + condition -> spatial + skip features
        self.dec_fc1 = nn.Linear(LATENT_SIZE + COND_SIZE, 512)
        self.dec_fc2 = nn.Linear(512, 256 * 13 * 6)

        # Skip feature decoders (skip recon from latent to prevent collaps)
        self.skip_dec_fc = nn.ModuleList([
            nn.Linear(LATENT_SIZE + COND_SIZE, s) for s in self.skip_sizes
        ])
        # Start with weaker skip contribution. model can increase it during training.
        self.skip_scale = nn.Parameter(torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32))

        # Upsampling
        self.dec_up3 = UpsampleBlock(256, 128)        # (B, 256, 13, 6) -> (B, 128, 26, 12)
        self.dec_up2 = UpsampleBlock(256, 64, output_padding=(0, 1))  # (B, 128+128, 26, 12) -> (B, 64, 52, 25)
        self.dec_up1 = UpsampleBlock(128, 32)         # (B, 64+64, 52, 25) -> (B, 32, 104, 50)

        # Final output (after skip concat: 32+32=64 channels)
        self.dec_final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 23, 3, padding=1),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def encode(self, x, c):
        """Encode spatial input to latent distribution (condition-independent encoder)."""
        # x: (B, 23, 104, 50), c: (B, 8)
        e0 = self.enc_conv0(x)                         # (B, 32, 104, 50)
        e1 = self.enc_down1(e0)                        # (B, 64, 52, 25)
        e2 = self.enc_down2(e1)                        # (B, 128, 26, 12)
        e3 = self.enc_down3(e2)                        # (B, 256, 13, 6)

        # Compact skip embeddings keep latent conditioning stable.
        skip0 = F.adaptive_avg_pool2d(e0, output_size=1).flatten(1)
        skip1 = F.adaptive_avg_pool2d(e1, output_size=1).flatten(1)
        skip2 = F.adaptive_avg_pool2d(e2, output_size=1).flatten(1)
        skips = [skip0, skip1, skip2]

        e3_flat = e3.view(e3.shape[0], -1)
        bottleneck = torch.cat([e3_flat] + skips + [c], dim=1)
        h = self.act(self.bottleneck_fc1(bottleneck))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(-10, 4)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, z, c):
        """Decode latent + condition to spatial output with skip connections przez latent space."""
        # z: (B, 128), c: (B, 8)
        h = torch.cat([z, c], dim=1)                   # (B, 136)
        h = self.act(self.dec_fc1(h))                  # (B, 512)
        h = self.dec_fc2(h)                            # (B, 256*13*6)
        h = h.view(-1, 256, 13, 6)                     # (B, 256, 13, 6)

        # skip features from latent (z, c)
        skip_feats = []
        zc = torch.cat([z, c], dim=1)
        for i, fc in enumerate(self.skip_dec_fc):
            s = fc(zc)
            s = s.view(-1, *self.skip_shapes[i])
            skip_feats.append(self.skip_scale[i] * s)

        # U-Net style skip connections, but coded in latent space (it prevents posterior collaps)
        # If skip connections werent coded in latent VAE would learn to extract inf from encoder (collaps)
        h = self.dec_up3(h)                        # (B, 128, 26, 12)
        h = torch.cat([h, skip_feats[2]], dim=1)   # (B, 256, 26, 12)
        h = self.dec_up2(h)                        # (B, 64, 52, 25)
        h = torch.cat([h, skip_feats[1]], dim=1)   # (B, 128, 52, 25)
        h = self.dec_up1(h)                        # (B, 32, 104, 50)
        h = torch.cat([h, skip_feats[0]], dim=1)   # (B, 64, 104, 50)
        h = self.dec_final(h)                     # (B, 23, 104, 50)
        return h

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)  # Uses stored skips
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train_epoch(model, optimizer, epoch):
    # Set model to training mode
    model.train()
    total = 0.0
    # Linearly anneal physics loss weight over PHYS_WARMUP epochs
    beta_phys = min(1.0, epoch / PHYS_WARMUP)
    # Progress bar for batch iteration
    pbar = tqdm(train_loader, desc='train', leave=False, unit='batch')
    for batch in pbar:
        # Prepare batch data
        x0, c = prepare_batch(batch)
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass: encode and decode
        recon, mu, logvar = model(x0, c)
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        # Reconstruction loss
        mse  = LAMBDA_RECON * F.mse_loss(recon, x0)

        # KL divergence loss per dimension
        kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kld  = kld_per_dim.sum(dim=1).mean()

        # Compute auxiliary physics losses
        aux, e, cont, ns, gb = aux_losses(recon, x0)

        # KL divergence with free bits threshold (it's pretty much important because it easily collapse)
        kld_fb_per_dim = torch.clamp(kld_per_dim, min=FREE_BITS)
        kld_fb = kld_fb_per_dim.sum(dim=1).mean()
 
        # Total loss: reconstruction + KLD + annealed physics loss
        kld_loss = BETA_KLD * kld_fb
        loss = mse + kld_loss + beta_phys * aux
  
        # Backward pass and optimization step
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        # Update progress bar with metrics
        pbar.set_postfix(
            loss=f'{loss.item()/x0.shape[0]:.5f}',
            mse=f'{mse.item():.4f}',
            kld=f'{kld.item():.3f}',
            E=f'{e.item():.3f}',
            cont=f'{cont.item():.3f}',
            ns=f'{ns.item():.3f}',
            gb=f'{gb.item():.3f}',
        )
    # Return average loss per sample
    return total / len(train_loader.dataset)


@torch.no_grad()
def evaluate(model):
    """Evaluate model performance on test set by computing MSE over multiple stochastic samples.
    
    For each test batch, generates N_EVAL latent samples and reconstructs the corresponding outputs,
    then averages the MSE across all samples and normalizes by field dimensions.
    """
    # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    
    # Accumulator for total MSE across all test batches
    total_mse = 0.0
    
    # Total number of field elements: 23 channels × NX × NY spatial dimensions
    field_elements = 23 * NX * NY  # (channels, height, width)
    
    # Iterate over test batches with progress bar
    for batch in tqdm(test_loader, desc='eval ', leave=False, unit='batch'):
        # Unpack and prepare batch: x0 is input tensor, c is condition vector
        x0, c = prepare_batch(batch)
        
        # Batch size
        B = x0.shape[0]
        
        # MSE accumulator for current batch across all N_EVAL samples
        mse = 0.0
        
        # Generate N_EVAL stochastic reconstructions per batch
        for _ in range(N_EVAL):
            # Sample latent vectors from standard normal distribution
            z     = torch.randn(B, LATENT_SIZE, device=device)
            
            # Decode latent vectors with condition to generate reconstruction
            recon = model.decode(z, c)
            
            # Accumulate MSE between reconstruction and ground truth (sum over all elements)
            mse  += F.mse_loss(recon, x0, reduction='sum').item()
        
        # Average MSE over N_EVAL samples and add to total
        total_mse += mse / N_EVAL
    
    # Return MSE normalized by total dataset size and field element dimensions
    return total_mse / (len(test_loader.dataset) * field_elements)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize model and move to specified device (GPU/CPU)
    model = PCVAE().to(device)
    
    # Create Adam optimizer with learning rate LR for all model parameters
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Setup cosine annealing learning rate scheduler that decays LR over all epochs
    # with minimum learning rate eta_min=1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Initialize best validation MSE to infinity for tracking best model checkpoint
    best_val = float('inf')
    
    # Create progress bar for training epochs
    epoch_bar = tqdm(range(1, args.epochs + 1), desc='PCVAE', unit='ep')
    
    # Main training loop over all epochs
    for epoch in epoch_bar:
        # Train for one epoch and get training loss
        train_loss = train_epoch(model, optimizer, epoch)
        
        # Evaluate model on validation set and get validation MSE
        val_mse    = evaluate(model)
        
        # Update learning rate according to scheduler
        scheduler.step()
        
        # Update progress bar with current training loss and validation MSE
        epoch_bar.set_postfix(train=f'{train_loss:.5f}', val=f'{val_mse:.6f}')

        # Check if current validation MSE is better than previous best
        if val_mse < best_val:
            # Update best validation MSE
            best_val = val_mse
            
            # Save model checkpoint with epoch number, validation MSE, model state and optimizer state
            torch.save({'epoch': epoch, 'val_mse': val_mse,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       CKPT_PATH)
            
            # Log best checkpoint save message
            tqdm.write(f'  [ep {epoch:>3}] val_mse={val_mse:.6f}  * best saved → {CKPT_PATH}')
        else:
            # Log current epoch validation MSE if not best
            tqdm.write(f'  [ep {epoch:>3}] val_mse={val_mse:.6f}')

    # Print training completion summary with best validation MSE and checkpoint location
    print(f'\nDone. best val_mse={best_val:.6f}  checkpoint: {CKPT_PATH}')

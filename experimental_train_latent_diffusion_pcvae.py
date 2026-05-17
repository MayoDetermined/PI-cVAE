# Experimental model !!!


"""
Train a latent diffusion prior for an existing PCVAE checkpoint.

Pipeline:
1) Freeze PCVAE encoder/decoder.
2) Encode training samples to latent z0 (mu or reparameterized sample).
3) Train a conditional DDPM denoiser in latent space z | c.
4) Sample latents with reverse diffusion and decode with PCVAE decoder.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--pcvae_checkpoint', type=str, default='train_PCVAE_results/best_PCVAE.pt')
parser.add_argument('--results_dir', type=str, default='train_LDM_results')
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--diff_steps', type=int, default=200)
parser.add_argument('--beta_start', type=float, default=1e-4)
parser.add_argument('--beta_end', type=float, default=2e-2)
parser.add_argument('--n_eval', type=int, default=1)
parser.add_argument('--latent_source', type=str, default='mu', choices=['mu', 'sample'])
parser.add_argument('--phys_warmup', type=int, default=30)
parser.add_argument('--lambda_energy', type=float, default=1e-3)
parser.add_argument('--lambda_continuity', type=float, default=1e-3)
parser.add_argument('--lambda_navier_stokes', type=float, default=1e-3)
parser.add_argument('--lambda_geometry_balance', type=float, default=5e-4)
parser.add_argument('--use_unet', action='store_true', help='Use U-Net denoiser instead of MLP')
parser.add_argument('--seed', type=int, default=42)


# Placeholder globals - will be initialized in if __name__ == '__main__'
args = None
device = None
loader_kwargs = None
train_loader = None
test_loader = None
CKPT_PATH = None

# ---------------------------------------------------------------------------
# Device / reproducibility
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LATENT_SIZE = 128
COND_SIZE = 8
NX, NY, NS = 104, 50, 10


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

_crx = torch.tensor(np.load(os.path.join('a_dataset', 'geometry', 'crx.npy')), dtype=torch.float32)
_cry = torch.tensor(np.load(os.path.join('a_dataset', 'geometry', 'cry.npy')), dtype=torch.float32)


def _cell_area_and_metrics(crx, cry):
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

_M_D = 2.0 * 1.67262192e-27
_M_N = 14.0 * 1.67262192e-27
_eV = 1.602176634e-19
_MASS = torch.tensor([_M_D, _M_D, _M_N, _M_N, _M_N, _M_N, _M_N, _M_N, _M_N, _M_N], dtype=torch.float32)
_CHARGE = torch.tensor([0, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)


def normalize_X(X):
    return (X - _X_min.to(X.device)) / (_X_max.to(X.device) - _X_min.to(X.device))


def _log_minmax(field, ln_min, ln_max):
    return (torch.log(field.clamp(min=1e-40)) - ln_min) / (ln_max - ln_min)


def normalize_fields(te, ti, na, ua, fnixap):
    te_n = _log_minmax(te, _te_ln_min, _te_ln_max).view(-1, NX * NY)
    ti_n = _log_minmax(ti, _ti_ln_min, _ti_ln_max).view(-1, NX * NY)

    na_ln = torch.log(na.clamp(min=1e-40))
    na_n = (na_ln - _na_ln_min.to(na.device)) / (_na_ln_max.to(na.device) - _na_ln_min.to(na.device))
    na_n = na_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)

    ua_n = (ua - _ua_min.to(ua.device)) / (_ua_max.to(ua.device) - _ua_min.to(ua.device))
    ua_n = ua_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)

    fnixap_n = _log_minmax(fnixap.view(-1, 1), _fnixap_ln_min, _fnixap_ln_max)
    fnixap_n = fnixap_n.expand(-1, NX * NY)

    return torch.cat([te_n, ti_n, na_n, ua_n, fnixap_n], dim=1)


def denormalize_fields(x_flat_or_spatial):
    if x_flat_or_spatial.dim() == 4:
        x_flat = x_flat_or_spatial.view(x_flat_or_spatial.shape[0], -1)
    else:
        x_flat = x_flat_or_spatial

    split = [NX * NY, NX * NY, NS * NX * NY, NS * NX * NY, NX * NY]
    te_n, ti_n, na_n, ua_n, fnixap_n = torch.split(x_flat, split, dim=1)

    te = torch.exp(te_n * (_te_ln_max - _te_ln_min) + _te_ln_min).view(-1, NX, NY)
    ti = torch.exp(ti_n * (_ti_ln_max - _ti_ln_min) + _ti_ln_min).view(-1, NX, NY)

    na_n = na_n.view(-1, NS, NX, NY).permute(0, 2, 3, 1)
    na = torch.exp(na_n * (_na_ln_max.to(x_flat.device) - _na_ln_min.to(x_flat.device))
                   + _na_ln_min.to(x_flat.device))

    ua_n = ua_n.view(-1, NS, NX, NY).permute(0, 2, 3, 1)
    ua = ua_n * (_ua_max.to(x_flat.device) - _ua_min.to(x_flat.device)) + _ua_min.to(x_flat.device)

    fnixap = torch.exp(fnixap_n * (_fnixap_ln_max - _fnixap_ln_min) + _fnixap_ln_min).view(-1)
    return te, ti, na, ua, fnixap


def _weighted_mse(pred, target, weight):
    w = weight.to(pred.device)
    if w.dim() == 2 and pred.dim() == 4:
        w = w[None, :, :, None]
    else:
        while w.dim() < pred.dim():
            w = w.unsqueeze(0)
    w = w.expand_as(pred)
    return ((pred - target).pow(2) * w).mean() / w.mean().clamp(min=1e-40)


def _metric_residual_2d(pred_r, pred_t, dx, dy, area_x, area_y):
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


def physics_loss_energy(recon, target):
    te_r, ti_r, na_r, ua_r, _ = denormalize_fields(recon)
    te_t, ti_t, na_t, ua_t, _ = denormalize_fields(target)

    mass = _MASS.to(recon.device)
    charge = _CHARGE.to(recon.device)

    n_e_r = (na_r * charge).sum(dim=-1)
    n_e_t = (na_t * charge).sum(dim=-1)
    n_i_r = na_r.sum(dim=-1)
    n_i_t = na_t.sum(dim=-1)

    kin_r = 0.5 * (na_r * ua_r.pow(2) * mass).sum(dim=-1)
    kin_t = 0.5 * (na_t * ua_t.pow(2) * mass).sum(dim=-1)

    e_r = 1.5 * (n_e_r * te_r + n_i_r * ti_r) * _eV + kin_r
    e_t = 1.5 * (n_e_t * te_t + n_i_t * ti_t) * _eV + kin_t
    return _metric_residual_2d(e_r, e_t, _DX_X, _DY_Y, _AREA_X, _AREA_Y)


def physics_loss_continuity(recon, target):
    _, _, na_r, ua_r, _ = denormalize_fields(recon)
    _, _, na_t, ua_t, _ = denormalize_fields(target)
    j_r = na_r * ua_r
    j_t = na_t * ua_t
    return _metric_residual_2d(j_r, j_t, _DX_X, _DY_Y, _AREA_X, _AREA_Y)


def physics_loss_navier_stokes(recon, target):
    _, ti_r, na_r, ua_r, _ = denormalize_fields(recon)
    _, ti_t, na_t, ua_t, _ = denormalize_fields(target)

    mass = _MASS.to(recon.device)
    mom_flux_r = na_r * ua_r.pow(2) * mass + na_r * ti_r.unsqueeze(-1) * _eV
    mom_flux_t = na_t * ua_t.pow(2) * mass + na_t * ti_t.unsqueeze(-1) * _eV

    return _metric_residual_2d(mom_flux_r, mom_flux_t, _DX_X, _DY_Y, _AREA_X, _AREA_Y)


def geometry_balance_loss(recon, target):
    te_r, ti_r, na_r, ua_r, _ = denormalize_fields(recon)
    te_t, ti_t, na_t, ua_t, _ = denormalize_fields(target)

    mass = _MASS.to(recon.device)
    charge = _CHARGE.to(recon.device)
    area = _CELL_AREA.to(recon.device)[None, :, :, None]

    e_r = (1.5 * (charge * na_r * te_r.unsqueeze(-1) * _eV + na_r * ti_r.unsqueeze(-1) * _eV)
           + 0.5 * na_r * ua_r.pow(2) * mass)
    e_t = (1.5 * (charge * na_t * te_t.unsqueeze(-1) * _eV + na_t * ti_t.unsqueeze(-1) * _eV)
           + 0.5 * na_t * ua_t.pow(2) * mass)

    j_r = na_r * ua_r
    j_t = na_t * ua_t
    mom_r = na_r * ua_r.pow(2) * mass + na_r * ti_r.unsqueeze(-1) * _eV
    mom_t = na_t * ua_t.pow(2) * mass + na_t * ti_t.unsqueeze(-1) * _eV

    e_int_r = (e_r * area).sum(dim=(1, 2))
    e_int_t = (e_t * area).sum(dim=(1, 2))
    j_int_r = (j_r * area).sum(dim=(1, 2))
    j_int_t = (j_t * area).sum(dim=(1, 2))
    mom_int_r = (mom_r * area).sum(dim=(1, 2))
    mom_int_t = (mom_t * area).sum(dim=(1, 2))

    scale_e = e_int_t.detach().abs().mean(dim=0, keepdim=True).clamp(min=1e-40)
    scale_j = j_int_t.detach().abs().mean(dim=0, keepdim=True).clamp(min=1e-40)
    scale_m = mom_int_t.detach().abs().mean(dim=0, keepdim=True).clamp(min=1e-40)

    loss_e = F.mse_loss(e_int_r / scale_e, e_int_t / scale_e)
    loss_j = F.mse_loss(j_int_r / scale_j, j_int_t / scale_j)
    loss_m = F.mse_loss(mom_int_r / scale_m, mom_int_t / scale_m)
    return (loss_e + loss_j + loss_m) / 3.0


def physics_aux_losses(recon, target):
    e = physics_loss_energy(recon, target)
    c = physics_loss_continuity(recon, target)
    ns = physics_loss_navier_stokes(recon, target)
    gb = geometry_balance_loss(recon, target)

    total = (
        args.lambda_energy * e
        + args.lambda_continuity * c
        + args.lambda_navier_stokes * ns
        + args.lambda_geometry_balance * gb
    )
    return total, e, c, ns, gb


def prepare_batch(batch):
    X, te, ti, na, ua, fnixap = [t.to(device) for t in batch]

    c = normalize_X(X)
    x0_flat = normalize_fields(te, ti, na, ua, fnixap).clamp(0.0, 1.0)

    B = x0_flat.shape[0]
    split = [NX * NY, NX * NY, NS * NX * NY, NS * NX * NY, NX * NY]
    te_n, ti_n, na_n, ua_n, fn_n = torch.split(x0_flat, split, dim=1)

    te_n = te_n.view(B, 1, NX, NY)
    ti_n = ti_n.view(B, 1, NX, NY)
    na_n = na_n.view(B, NS, NX, NY)
    ua_n = ua_n.view(B, NS, NX, NY)
    fn_n = fn_n.view(B, 1, NX, NY)

    x0 = torch.cat([te_n, ti_n, na_n, ua_n, fn_n], dim=1)
    return x0, c


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SimDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        base = os.path.join('a_dataset', split)

        self.X = torch.tensor(np.load(os.path.join(base, 'X_tmp.npy')), dtype=torch.float32)
        self.te = torch.tensor(np.load(os.path.join(base, 'te_tmp.npy')), dtype=torch.float32)
        self.ti = torch.tensor(np.load(os.path.join(base, 'ti_tmp.npy')), dtype=torch.float32)
        self.na = torch.tensor(np.load(os.path.join(base, 'na_tmp.npy')), dtype=torch.float32)
        self.ua = torch.tensor(np.load(os.path.join(base, 'ua_tmp.npy')), dtype=torch.float32)
        self.fnixap = torch.tensor(np.load(os.path.join(base, 'fnixap_tmp.npy')), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.te[idx], self.ti[idx], self.na[idx], self.ua[idx], self.fnixap[idx]



# ---------------------------------------------------------------------------
# Diffusion components
# ---------------------------------------------------------------------------
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class CrossAttention(nn.Module):
    def __init__(self, channels, cond_dim, num_heads=4, head_dim=64):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        
        self.to_q = nn.Conv2d(channels, inner_dim, 1)
        self.to_k = nn.Linear(cond_dim, inner_dim)
        self.to_v = nn.Linear(cond_dim, inner_dim)
        self.to_out = nn.Conv2d(inner_dim, channels, 1)
    
    def forward(self, x, cond):
        # x: (batch, channels, height, width)
        # cond: (batch, cond_dim)
        
        b, c, h, w = x.shape
        
        # Query from feature map
        q = self.to_q(x)  # (batch, inner_dim, h, w)
        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)  # (batch, heads, h*w, head_dim)
        
        # Key and Value from condition
        k = self.to_k(cond)  # (batch, inner_dim)
        v = self.to_v(cond)  # (batch, inner_dim)
        k = k.view(b, self.num_heads, self.head_dim)  # (batch, heads, head_dim)
        v = v.view(b, self.num_heads, self.head_dim)  # (batch, heads, head_dim)
        
        # Attention
        scale = self.head_dim ** -0.5
        sim = torch.einsum('b h n d, b h d -> b h n', q, k) * scale
        attn = sim.softmax(dim=-1)

        # Broadcast value vectors across spatial query positions and weight by attention.
        out = attn.unsqueeze(-1) * v.unsqueeze(2)
        out = out.transpose(2, 3).reshape(b, self.num_heads * self.head_dim, h, w)
        
        return self.to_out(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_ch, cond_ch):
        super().__init__()
        self.ln1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.ln2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.time_proj = nn.Linear(t_emb_ch, out_ch)
        self.cond_proj = nn.Linear(cond_ch, out_ch)
        
        self.cross_attn = CrossAttention(out_ch, cond_ch, num_heads=4)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb, cond_emb):
        h = F.gelu(self.ln1(x))
        h = self.conv1(h)
        
        t_emb_proj = self.time_proj(t_emb)
        cond_emb_proj = self.cond_proj(cond_emb)
        
        while t_emb_proj.dim() < h.dim():
            t_emb_proj = t_emb_proj.unsqueeze(-1)
        while cond_emb_proj.dim() < h.dim():
            cond_emb_proj = cond_emb_proj.unsqueeze(-1)
        
        h = h + t_emb_proj + cond_emb_proj
        
        # Cross-attention with condition
        h = self.cross_attn(h, cond_emb) + h
        
        h = F.gelu(self.ln2(h))
        h = self.conv2(h)
        
        return h + self.skip(x)


class UNetDenoiser(nn.Module):
    def __init__(self, latent_dim=LATENT_SIZE, cond_dim=COND_SIZE, t_dim=128, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.t_dim = t_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
        )
        
        self.spatial_size = 4
        self.init_channels = latent_dim // (self.spatial_size ** 2)
        
        self.init_conv = nn.Conv2d(self.init_channels, hidden, 3, padding=1)
        
        self.enc1 = ResBlock(hidden, hidden, t_dim, hidden)
        self.down1 = nn.Conv2d(hidden, hidden * 2, 4, stride=2, padding=1)
        
        self.enc2 = ResBlock(hidden * 2, hidden * 2, t_dim, hidden)
        
        self.up1 = nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1)
        self.dec1 = ResBlock(hidden * 2, hidden, t_dim, hidden)
        
        self.final_ln = nn.GroupNorm(8, hidden)
        self.final_conv = nn.Conv2d(hidden, self.init_channels, 3, padding=1)
        self.final_proj = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, z_t, t, c):
        batch_size = z_t.shape[0]
        
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.time_mlp(t_emb)
        
        c_emb = self.cond_mlp(c)
        
        x = z_t.view(batch_size, self.init_channels, self.spatial_size, self.spatial_size)
        x = self.init_conv(x)
        
        x = self.enc1(x, t_emb, c_emb)
        skip1 = x
        x = self.down1(x)
        
        x = self.enc2(x, t_emb, c_emb)
        
        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x, t_emb, c_emb)
        
        x = F.gelu(self.final_ln(x))
        x = self.final_conv(x)
        
        x = x.view(batch_size, self.latent_dim)
        x = self.final_proj(x)
        
        return x


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim=LATENT_SIZE, cond_dim=COND_SIZE, t_dim=128, hidden=512):
        super().__init__()
        self.t_dim = t_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim),
        )

        in_dim = latent_dim + cond_dim + t_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, latent_dim)
        self.act = nn.GELU()

    def forward(self, z_t, t, c):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.time_mlp(t_emb)

        h = torch.cat([z_t, c, t_emb], dim=1)
        h1 = self.act(self.fc1(h))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2 + h1))
        return self.fc4(h3)


class DiffusionSchedule:
    def __init__(self, num_steps, beta_start, beta_end, device):
        self.num_steps = num_steps

        # Cosine schedule
        t = torch.arange(num_steps, dtype=torch.float32, device=device) / num_steps
        betas = 0.5 * (1 - torch.cos(np.pi * t))
        betas = betas * (beta_end - beta_start) + beta_start
        betas = betas.clamp(min=1e-4, max=0.999)
        
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.posterior_var = posterior_var.clamp(min=1e-20)

    def _extract(self, arr, t, x_shape):
        out = arr.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, z0, t, noise):
        s1 = self._extract(self.sqrt_alpha_bars, t, z0.shape)
        s2 = self._extract(self.sqrt_one_minus_alpha_bars, t, z0.shape)
        return s1 * z0 + s2 * noise


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
# Train / eval
# ---------------------------------------------------------------------------
def get_latent_targets(vae, x0, c):
    with torch.no_grad():
        mu, logvar = vae.encode(x0, c)
        if args.latent_source == 'sample':
            z0 = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z0 = mu
    return z0


def train_epoch(vae, denoiser, schedule, optimizer, epoch):
    denoiser.train()
    total = 0.0
    total_diff = 0.0
    total_phys = 0.0

    beta_phys = min(1.0, epoch / max(1, args.phys_warmup))

    pbar = tqdm(train_loader, desc='ldm-train', leave=False, unit='batch')
    for batch in pbar:
        x0, c = prepare_batch(batch)
        z0 = get_latent_targets(vae, x0, c)

        t = torch.randint(0, schedule.num_steps, (z0.shape[0],), device=device)
        noise = torch.randn_like(z0)
        z_t = schedule.q_sample(z0, t, noise)

        pred_noise = denoiser(z_t, t, c)
        diff_loss = F.mse_loss(pred_noise, noise)

        sqrt_abar_t = schedule._extract(schedule.sqrt_alpha_bars, t, z_t.shape)
        sqrt_one_minus_abar_t = schedule._extract(schedule.sqrt_one_minus_alpha_bars, t, z_t.shape)
        z0_pred = (z_t - sqrt_one_minus_abar_t * pred_noise) / sqrt_abar_t.clamp(min=1e-12)

        recon_pred = vae.decode(z0_pred, c).clamp(0.0, 1.0)
        phys_loss, e, cont, ns, gb = physics_aux_losses(recon_pred, x0)

        loss = diff_loss + beta_phys * phys_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()

        total += loss.item() * z0.shape[0]
        total_diff += diff_loss.item() * z0.shape[0]
        total_phys += phys_loss.item() * z0.shape[0]
        pbar.set_postfix(
            loss=f'{loss.item():.6f}',
            diff=f'{diff_loss.item():.6f}',
            phys=f'{phys_loss.item():.6f}',
            E=f'{e.item():.3f}',
            cont=f'{cont.item():.3f}',
            ns=f'{ns.item():.3f}',
            gb=f'{gb.item():.3f}',
        )

    n = len(train_loader.dataset)
    return total / n, total_diff / n, total_phys / n


@torch.no_grad()
def evaluate(vae, denoiser, schedule):
    denoiser.eval()
    total = 0.0
    total_diff = 0.0
    total_phys = 0.0
    total_mse = 0.0
    field_elements = 23 * NX * NY

    for batch in tqdm(test_loader, desc='ldm-eval ', leave=False, unit='batch'):
        x0, c = prepare_batch(batch)
        z0 = get_latent_targets(vae, x0, c)

        t = torch.randint(0, schedule.num_steps, (z0.shape[0],), device=device)
        noise = torch.randn_like(z0)
        z_t = schedule.q_sample(z0, t, noise)
        pred_noise = denoiser(z_t, t, c)

        sqrt_abar_t = schedule._extract(schedule.sqrt_alpha_bars, t, z_t.shape)
        sqrt_one_minus_abar_t = schedule._extract(schedule.sqrt_one_minus_alpha_bars, t, z_t.shape)
        z0_pred = (z_t - sqrt_one_minus_abar_t * pred_noise) / sqrt_abar_t.clamp(min=1e-12)
        recon_pred = vae.decode(z0_pred, c).clamp(0.0, 1.0)

        diff_loss = F.mse_loss(pred_noise, noise)
        phys_loss, _, _, _, _ = physics_aux_losses(recon_pred, x0)
        loss = diff_loss + phys_loss

        total += loss.item() * z0.shape[0]
        total_diff += diff_loss.item() * z0.shape[0]
        total_phys += phys_loss.item() * z0.shape[0]

        mse_acc = 0.0
        for _ in range(args.n_eval):
            z_gen = sample_latents(denoiser, schedule, c)
            recon = vae.decode(z_gen, c)
            mse_acc += F.mse_loss(recon, x0, reduction='sum').item()

        total_mse += mse_acc / args.n_eval

    n = len(test_loader.dataset)
    mean_total = total / n
    mean_diff = total_diff / n
    mean_phys = total_phys / n
    mean_mse = total_mse / (len(test_loader.dataset) * field_elements)
    return mean_total, mean_diff, mean_phys, mean_mse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    CKPT_PATH = os.path.join(args.results_dir, 'best_latent_diffusion.pt')
    
    # Device / reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_kwargs = {'num_workers': 0, 'pin_memory': torch.cuda.is_available()}
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f'Device: {device}')
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        SimDataset('train'), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(
        SimDataset('test'), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    
    from main_train_pcvae import PCVAE
    
    vae_ckpt = load_checkpoint(args.pcvae_checkpoint, device)

    vae = PCVAE().to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()

    for p in vae.parameters():
        p.requires_grad = False

    denoiser = UNetDenoiser().to(device) if args.use_unet else LatentDenoiser().to(device)
    schedule = DiffusionSchedule(
        num_steps=args.diff_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    optimizer = optim.AdamW(denoiser.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_mse = float('inf')

    print(f"Loaded PCVAE checkpoint: {args.pcvae_checkpoint}")
    print(f"Training latent diffusion in z-space ({LATENT_SIZE}D), source={args.latent_source}")
    print(f"Denoiser: {'U-Net' if args.use_unet else 'MLP'}")
    print(
        'Physics weights: '
        f"E={args.lambda_energy}, cont={args.lambda_continuity}, "
        f"NS={args.lambda_navier_stokes}, GB={args.lambda_geometry_balance}, "
        f"warmup={args.phys_warmup}"
    )

    epoch_bar = tqdm(range(1, args.epochs + 1), desc='LDM', unit='ep')
    for epoch in epoch_bar:
        train_loss, train_diff, train_phys = train_epoch(vae, denoiser, schedule, optimizer, epoch)
        val_loss, val_diff, val_phys, val_mse = evaluate(vae, denoiser, schedule)

        scheduler.step()
        epoch_bar.set_postfix(
            train=f'{train_loss:.6f}',
            train_diff=f'{train_diff:.6f}',
            train_phys=f'{train_phys:.6f}',
            val=f'{val_loss:.6f}',
            val_diff=f'{val_diff:.6f}',
            val_phys=f'{val_phys:.6f}',
            val_mse=f'{val_mse:.6f}',
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(
                {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_diff': val_diff,
                    'val_phys': val_phys,
                    'val_mse': val_mse,
                    'denoiser_state_dict': denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'pcvae_checkpoint': args.pcvae_checkpoint,
                    'latent_source': args.latent_source,
                    'diff_steps': args.diff_steps,
                    'beta_start': args.beta_start,
                    'beta_end': args.beta_end,
                    'phys_warmup': args.phys_warmup,
                    'lambda_energy': args.lambda_energy,
                    'lambda_continuity': args.lambda_continuity,
                    'lambda_navier_stokes': args.lambda_navier_stokes,
                    'lambda_geometry_balance': args.lambda_geometry_balance,
                },
                CKPT_PATH,
            )
            tqdm.write(
                f'  [ep {epoch:>3}] val={val_loss:.6f}  val_diff={val_diff:.6f}  '
                f'val_phys={val_phys:.6f}  val_mse={val_mse:.6f}  * best saved -> {CKPT_PATH}'
            )
        else:
            tqdm.write(
                f'  [ep {epoch:>3}] val={val_loss:.6f}  val_diff={val_diff:.6f}  '
                f'val_phys={val_phys:.6f}  val_mse={val_mse:.6f}'
            )

    print(f'\nDone. best val_mse={best_val_mse:.6f}  checkpoint: {CKPT_PATH}')

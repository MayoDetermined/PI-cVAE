"""
Training loop for the ConditionalVAE physics model.

Model input:  stacked fields (te, ti, na, ua) as 22 channels -> (batch, 22, 104, 50)
Conditioning: 8 input parameters X -> (batch, 8)

Channel layout of stacked tensor:
  channel 0    : te  (electron temperature)
  channel 1    : ti  (ion temperature)
  channels 2-11: na  (10 species densities)
  channels 12-21: ua (10 species velocities)

Losses:
  - Reconstruction MSE   (normalized space)
  - VAE KL regularisation
  - Energy conservation  (physics constraint, small weight)
  - Momentum conservation (physics constraint, small weight)
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data_handler import PlasmaDataHandler
from physics_model.model import ConditionalVAE
from physics_model.loss.energy import EnergyConservationLoss
from physics_model.loss.momentum import MomentumConservationLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Torch-native denormalizer (gradient-preserving)
# ---------------------------------------------------------------------------

class TorchDenormalizer:
    """
    Reverses min-max normalization using torch operations (gradient-preserving).

    Mirrors PlasmaDataHandler.denormalize():
      - te, ti, na : norm_stats_minmax stores LN values in min/max → exp() after linear interp
      - ua         : arcsinh reverse transform
      - others     : linear only

    All stat arrays are cached as float32 tensors on the target device.
    """

    def __init__(self, norm_stats: dict):
        self._stats = norm_stats        # raw numpy dict from handler
        self._cache: dict = {}          # device -> field -> tensors

    def _get(self, field: str, device: torch.device):
        key = (field, device)
        if key not in self._cache:
            vmin = torch.tensor(self._stats[f"{field}_min"], dtype=torch.float32, device=device)
            vmax = torch.tensor(self._stats[f"{field}_max"], dtype=torch.float32, device=device)
            self._cache[key] = (vmin, vmax)
        return self._cache[key]

    def denorm(self, field: str, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor ``x`` (values should be in [0, 1]) for the given field.

        Mirrors PlasmaDataHandler.denormalize() exactly:
        - te/ti/na: stats store LN values directly in min/max → linear interp then exp()
        - ua: arcsinh reverse transform for bipolar velocities
        - X, others: linear denormalization
        """
        vmin, vmax = self._get(field, x.device)

        if field == "ua":
            # ua: reverse arcsinh transform
            # min=1.0 mirrors PlasmaDataHandler.denormalize (np.maximum(scale, 1.0))
            scale = torch.maximum(torch.abs(vmin), torch.abs(vmax))
            scale = torch.clamp(scale, min=1.0)
            asinh_max = torch.asinh(torch.tensor(1.0, dtype=torch.float32, device=x.device))
            # Reverse: [0,1] → [-asinh_max, asinh_max] → sinh → original scale
            data_arcsinh = x * (2.0 * asinh_max) - asinh_max
            out = scale * torch.sinh(data_arcsinh)
            return out

        out = x * (vmax - vmin) + vmin

        if field in ("te", "ti", "na"):
            # norm_stats_minmax.npz stores natural-log values directly in min/max
            # (same convention as data_handler.denormalize).  Clamp before exp()
            # to prevent float32 overflow from out-of-range model outputs.
            # For te/ti: vmin/vmax are scalar; for na: shape (10,) — clamp per-species.
            if vmin.numel() == 1:
                out = torch.clamp(out, min=vmin.item(), max=vmax.item())
            else:
                # torch.clamp supports tensor min/max (PyTorch >= 1.9), broadcasts
                # against the last dim of out (..., 10)
                out = torch.clamp(out, min=vmin, max=vmax)
            out = torch.exp(out)

        return out


# ---------------------------------------------------------------------------
# Warmup scheduling
# ---------------------------------------------------------------------------

def compute_physics_weight_warmup(epoch: int, warmup_epochs: int, weight_init: float, weight_end: float) -> float:
    """
    Logarithmically increase physics loss weight over warmup_epochs.
    
    Maps from weight_init (epoch 0) to weight_end (epoch warmup_epochs).
    
    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of epochs for warmup phase
        weight_init: Initial physics weight at epoch 0
        weight_end: Target physics weight after warmup
    
    Returns:
        Current physics weight
    """
    if warmup_epochs <= 0:
        return weight_end
    
    # Ensure positive weights for log scale
    import math
    weight_init = max(weight_init, 1e-30)
    weight_end = max(weight_end, 1e-30)
    
    progress = min(1.0, (epoch + 1) / warmup_epochs)
    
    # Logarithmic scale: log(w) = log(init) + progress * (log(end) - log(init))
    log_weight = math.log10(weight_init) + progress * (math.log10(weight_end) - math.log10(weight_init))
    weight = 10.0 ** log_weight
    
    return weight


# ---------------------------------------------------------------------------
# Pearson correlation loss helper (targets ua spatial structure)
# ---------------------------------------------------------------------------

def ua_spatial_corr_loss(pred: torch.Tensor, gt: torch.Tensor,
                          var_threshold: float = 1e-4) -> torch.Tensor:
    """
    1 - mean Pearson correlation between predicted and GT ua channels,
    computed spatially (flattened H×W) per (batch, channel) pair.

    Channels where GT has near-zero spatial variance (e.g. species with almost
    no flow, GT ≈ 0.5 everywhere) are **excluded** from the mean — their
    correlation is undefined and including them would inject noise into the
    gradient.

    Args:
        pred: (B, 10, H, W)  — predicted ua channels
        gt:   (B, 10, H, W)  — ground-truth ua channels
        var_threshold: minimum GT spatial std (in normalised [0,1] units)
                       below which a (batch, channel) pair is skipped.
                       Default 1e-4 corresponds to ~0.01 % of the full range.

    Returns:
        scalar loss in [0, 2]; 0 = perfect correlation, 2 = perfect
        anti-correlation.  Returns 0.0 (no gradient) if all channels are
        below the variance threshold.
    """
    B, C, H, W = pred.shape
    p = pred.reshape(B * C, -1)   # (B*C, H*W)
    g = gt.reshape(B * C, -1)

    p_c = p - p.mean(dim=-1, keepdim=True)
    g_c = g - g.mean(dim=-1, keepdim=True)

    g_std = g_c.norm(dim=-1)                    # (B*C,) — 0 when GT is flat
    active = g_std > var_threshold * (H * W) ** 0.5  # normalise by sqrt(n_pixels)

    if not active.any():
        return pred.sum() * 0.0                 # zero loss with gradient path

    p_c_a = p_c[active]
    g_c_a = g_c[active]
    g_std_a = g_std[active]
    p_std_a = p_c_a.norm(dim=-1).clamp(min=1e-8)

    corr = (p_c_a * g_c_a).sum(dim=-1) / (p_std_a * g_std_a)  # (n_active,)
    return 1.0 - corr.mean()                    # 0 = perfect


# ---------------------------------------------------------------------------
# cVAE KL divergence helper
# ---------------------------------------------------------------------------

def compute_cvae_kl(model: "ConditionalVAE", mu: torch.Tensor, logvar: torch.Tensor,
                    cond: torch.Tensor, kl_weight: float,
                    free_bits: float = 0.0) -> tuple:
    """
    Compute KL divergence for cVAE or standard VAE.

    cVAE (use_prior_net=True):  KL( q(z|x,c) || p(z|c) )
      = 0.5 * sum[ logvar_p - logvar_q + (exp(logvar_q) + (mu_q-mu_p)^2) / exp(logvar_p) - 1 ]

    Standard VAE:               KL( q(z|x) || N(0,I) )
      = -0.5 * sum[ 1 + logvar - mu^2 - exp(logvar) ]

    free_bits > 0: each latent dimension contributes at least ``free_bits`` to
    the KL (per sample), preventing posterior collapse where the encoder ignores
    its input and collapses to the prior.

    Returns:
        (loss, n_active_dims)  where n_active_dims is the number of latent dims
        with mean KL above free_bits (useful for diagnosing collapse).
    """
    if model.use_prior_net:
        mu_p, logvar_p = model.prior_net(cond)
        # Clamp logvars to prevent exp() overflow / division by near-zero
        logvar   = torch.clamp(logvar,   -10.0, 10.0)
        logvar_p = torch.clamp(logvar_p, -10.0, 10.0)
        # KL between two diagonal Gaussians: shape (batch, latent_dim)
        kl = 0.5 * (
            logvar_p - logvar
            + (logvar.exp() + (mu - mu_p).pow(2)) / logvar_p.exp()
            - 1
        )
    else:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Count active dimensions (mean KL > free_bits threshold) before clamping
    kl_per_dim = kl.mean(dim=0)          # (latent_dim,)
    n_active = int((kl_per_dim > max(free_bits, 0.0)).sum().item())

    # Free-bits: floor each latent dimension's KL contribution, so no dimension
    # can be "turned off" without cost.  Applied per-dim before summing.
    if free_bits > 0.0:
        kl = kl.clamp(min=free_bits)

    return kl_weight * kl.sum(dim=1).mean(), n_active


def compute_prior_kl(model: "ConditionalVAE", cond: torch.Tensor,
                     prior_kl_weight: float) -> torch.Tensor:
    """
    KL( p(z|c) || N(0,I) )  — regularises the prior network.

    Prevents the learned prior from drifting arbitrarily far from standard
    normal, keeping the latent space well-shaped for unconditional sampling
    and improving generalisation of the prior net.

    Only meaningful when use_prior_net=True; returns 0 otherwise.

    KL(N(mu_p, diag(sigma_p^2)) || N(0,I))
        = 0.5 * sum[ mu_p^2 + exp(logvar_p) - logvar_p - 1 ]
    """
    if not model.use_prior_net or prior_kl_weight == 0.0:
        return torch.tensor(0.0, device=cond.device)
    mu_p, logvar_p = model.prior_net(cond)
    logvar_p = torch.clamp(logvar_p, -10.0, 10.0)
    kl = 0.5 * (mu_p.pow(2) + logvar_p.exp() - logvar_p - 1.0)
    return prior_kl_weight * kl.sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlasmaDataset(Dataset):
    """
    Wraps PlasmaDataHandler to expose (stacked_fields, X) pairs.

    All fields are min-max normalised before returning.
    Stacking order: te(1) | ti(1) | na(10) | ua(10)  => 22 channels.
    """

    def __init__(self, handler: PlasmaDataHandler):
        self.handler = handler
        self.n = handler.split_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        h = self.handler

        # Raw arrays for this sample
        te = h.te[idx]       # (104, 50)
        if h.ti is None:
            raise RuntimeError(
                "ti (ion temperature) not loaded — ti_tmp.npy missing in split directory. "
                "This model requires all 22 channels including ti."
            )
        ti = h.ti[idx]       # (104, 50)
        na = h.na[idx]       # (104, 50, 10)
        ua = h.ua[idx]       # (104, 50, 10)
        X  = h.X[idx]        # (8,)

        # Normalise
        te_n = h.normalize("te", te[None])[0]          # keep (104,50)
        ti_n = h.normalize("ti", ti[None])[0]
        na_n = h.normalize("na", na[None])[0]          # (104,50,10)
        ua_n = h.normalize("ua", ua[None])[0]
        X_n  = h.normalize("X",  X[None])[0]           # (8,)

        # Stack into (22, 104, 50)
        te_t = torch.tensor(te_n, dtype=torch.float32).unsqueeze(0)   # (1,104,50)
        ti_t = torch.tensor(ti_n, dtype=torch.float32).unsqueeze(0)
        na_t = torch.tensor(na_n, dtype=torch.float32).permute(2,0,1) # (10,104,50)
        ua_t = torch.tensor(ua_n, dtype=torch.float32).permute(2,0,1)

        fields = torch.cat([te_t, ti_t, na_t, ua_t], dim=0)           # (22,104,50)
        cond   = torch.tensor(X_n,  dtype=torch.float32)               # (8,)

        return fields, cond


# ---------------------------------------------------------------------------
# Helper: unpack 22-channel tensor into separate physical fields
# ---------------------------------------------------------------------------

def unpack_fields(x: torch.Tensor):
    """
    Split (batch, 22, nx, ny) into te, ti, na, ua.

    Returns:
        te : (batch, nx, ny)
        ti : (batch, nx, ny)
        na : (batch, nx, ny, 10)
        ua : (batch, nx, ny, 10)
    """
    te = x[:, 0]                              # (B, nx, ny)
    ti = x[:, 1]
    na = x[:, 2:12].permute(0, 2, 3, 1)      # (B, nx, ny, 10)
    ua = x[:, 12:22].permute(0, 2, 3, 1)
    return te, ti, na, ua


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def denorm_fields(decoded: torch.Tensor, fields: torch.Tensor, denorm: TorchDenormalizer):
    """
    Denormalize 22-channel tensors into individual physical fields.

    Returns six tensors (pred and true) with real physical units,
    suitable for physics loss functions.
    """
    te_p_n, ti_p_n, na_p_n, ua_p_n = unpack_fields(decoded)
    te_t_n, ti_t_n, na_t_n, ua_t_n = unpack_fields(fields)

    te_pred = denorm.denorm("te", te_p_n)
    ti_pred = denorm.denorm("ti", ti_p_n)
    na_pred = denorm.denorm("na", na_p_n)
    ua_pred = denorm.denorm("ua", ua_p_n)

    with torch.no_grad():
        te_true = denorm.denorm("te", te_t_n)
        ti_true = denorm.denorm("ti", ti_t_n)
        na_true = denorm.denorm("na", na_t_n)
        ua_true = denorm.denorm("ua", ua_t_n)

    return te_pred, ti_pred, na_pred, ua_pred, te_true, ti_true, na_true, ua_true


def train_epoch(
    model: ConditionalVAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fns: dict,
    device: torch.device,
    physics_weight: float,
) -> dict:
    model.train()
    totals = {"recon": 0.0, "kl": 0.0, "prior_kl": 0.0, "energy": 0.0, "momentum": 0.0, "total": 0.0,
              "mse_te": 0.0, "mse_ti": 0.0, "mse_na": 0.0, "mse_ua": 0.0}
    n_batches = 0
    total_active_dims = 0

    for batch_idx, (fields, cond) in enumerate(loader):
        fields = fields.to(device)
        cond   = cond.to(device)

        if batch_idx == 0:
            logger.info(f"Input fields range: [{fields.min():.4f}, {fields.max():.4f}]")
            logger.info(f"Input cond range: [{cond.min():.4f}, {cond.max():.4f}]")
            # Check for out-of-range normalized values
            fields_clamped = torch.clamp(fields, -1.0, 2.0)
            if not torch.equal(fields, fields_clamped):
                logger.warning(f"Fields out of expected [0, 1] range (possibly slightly outside)")

        decoded, mu, logvar, _ = model(fields, cond)
        
        # Debug: check decoded ranges
        if batch_idx == 0:
            logger.info(f"Decoded range: [{decoded.min():.4f}, {decoded.max():.4f}]")
            # Clamp decoded for safety
            if decoded.min() < -2.0 or decoded.max() > 2.5:
                logger.warning(f"Decoded values very large, clamping may be needed")

        # 1. Reconstruction loss — per-group weighted MSE so that ua channels
        # receive ua_recon_weight x more gradient than te/ti/na channels.
        # Normalised so total scale is equivalent to plain mse_loss when weight=1.
        ua_w    = loss_fns.get("ua_recon_weight", 1.0)
        ua_cw   = loss_fns.get("ua_corr_weight",  0.0)
        loss_recon_tena = nn.functional.mse_loss(decoded[:, :12],  fields[:, :12])
        loss_recon_ua   = nn.functional.mse_loss(decoded[:, 12:22], fields[:, 12:22])
        loss_recon = (12 * loss_recon_tena + ua_w * 10 * loss_recon_ua) / (12 + ua_w * 10)

        # Correlation loss for ua: penalises missing spatial structure
        # (low RMSE but flat prediction → corr≈0 → this term stays ~1.0)
        if ua_cw > 0.0:
            loss_recon = loss_recon + ua_cw * ua_spatial_corr_loss(
                decoded[:, 12:22], fields[:, 12:22]
            )

        # Per-group MSE (detached — logging only, does not affect backprop)
        mse_te_b = nn.functional.mse_loss(decoded[:, 0:1].detach(),   fields[:, 0:1]).item()
        mse_ti_b = nn.functional.mse_loss(decoded[:, 1:2].detach(),   fields[:, 1:2]).item()
        mse_na_b = nn.functional.mse_loss(decoded[:, 2:12].detach(),  fields[:, 2:12]).item()
        mse_ua_b = nn.functional.mse_loss(decoded[:, 12:22].detach(), fields[:, 12:22]).item()

        # 2. VAE / cVAE KL regularisation
        loss_kl, n_active = compute_cvae_kl(
            model, mu, logvar, cond,
            loss_fns["kl_weight"], loss_fns.get("free_bits", 0.0),
        )
        total_active_dims += n_active

        # 2b. Prior regularisation — KL( p(z|c) || N(0,I) )
        loss_prior_kl = compute_prior_kl(model, cond, loss_fns.get("prior_kl_weight", 0.0))

        # 3. Physics losses on denormalised (physical-unit) values
        if physics_weight > 0:
            te_pred, ti_pred, na_pred, ua_pred, \
            te_true, ti_true, na_true, ua_true = denorm_fields(decoded, fields, loss_fns["denorm"])
            
            if batch_idx == 0:
                logger.info(f"Denorm te_pred range: [{te_pred.min():.4e}, {te_pred.max():.4e}]")
                logger.info(f"Denorm na_pred range: [{na_pred.min():.4e}, {na_pred.max():.4e}]")

            energy_dict = loss_fns["energy"].forward(
                te_pred=te_pred, ti_pred=ti_pred,
                na_pred=na_pred, ua_pred=ua_pred,
                te_true=te_true, ti_true=ti_true,
                na_true=na_true, ua_true=ua_true,
            )
            loss_energy = energy_dict["loss_total"]
            # Clamp to guard against occasional numerical spikes in physics losses

            momentum_dict = loss_fns["momentum"].forward(
                na_pred=na_pred, ua_pred=ua_pred, te_pred=te_pred,
                na_true=na_true, ua_true=ua_true,
            )
            loss_momentum = momentum_dict["loss_total"]
        else:
            loss_energy = torch.tensor(0.0, device=device)
            loss_momentum = torch.tensor(0.0, device=device)

        loss_total = (
            loss_recon
            + loss_kl
            + loss_prior_kl
            + physics_weight * loss_energy
            + physics_weight * loss_momentum
        )

        optimizer.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["recon"]     += loss_recon.item()
        totals["kl"]        += loss_kl.item()
        totals["prior_kl"]  += loss_prior_kl.item() if isinstance(loss_prior_kl, torch.Tensor) else loss_prior_kl
        totals["energy"]    += loss_energy.item() if isinstance(loss_energy, torch.Tensor) else loss_energy
        totals["momentum"]  += loss_momentum.item() if isinstance(loss_momentum, torch.Tensor) else loss_momentum
        totals["total"]     += loss_total.item()
        totals["mse_te"]   += mse_te_b
        totals["mse_ti"]   += mse_ti_b
        totals["mse_na"]   += mse_na_b
        totals["mse_ua"]   += mse_ua_b
        n_batches += 1

    results = {k: v / n_batches for k, v in totals.items()}
    results["active_dims"] = total_active_dims / n_batches
    return results


@torch.no_grad()
def val_epoch(
    model: ConditionalVAE,
    loader: DataLoader,
    loss_fns: dict,
    device: torch.device,
    physics_weight: float,
) -> dict:
    model.eval()
    totals = {"recon": 0.0, "kl": 0.0, "prior_kl": 0.0, "energy": 0.0, "momentum": 0.0, "total": 0.0,
              "mse_te": 0.0, "mse_ti": 0.0, "mse_na": 0.0, "mse_ua": 0.0}
    n_batches = 0

    for fields, cond in loader:
        fields = fields.to(device)
        cond   = cond.to(device)

        decoded, mu, logvar, _ = model(fields, cond)

        ua_w  = loss_fns.get("ua_recon_weight", 1.0)
        ua_cw = loss_fns.get("ua_corr_weight",  0.0)
        loss_recon_tena = nn.functional.mse_loss(decoded[:, :12],  fields[:, :12])
        loss_recon_ua   = nn.functional.mse_loss(decoded[:, 12:22], fields[:, 12:22])
        # Pure weighted MSE — tracked in totals["recon"] for best checkpoint selection
        loss_recon_mse = (12 * loss_recon_tena + ua_w * 10 * loss_recon_ua) / (12 + ua_w * 10)
        # Add correlation loss for total (does NOT affect best_val_recon tracking)
        loss_recon = loss_recon_mse
        if ua_cw > 0.0:
            loss_recon = loss_recon + ua_cw * ua_spatial_corr_loss(
                decoded[:, 12:22], fields[:, 12:22]
            )
        mse_te_b = nn.functional.mse_loss(decoded[:, 0:1],   fields[:, 0:1]).item()
        mse_ti_b = nn.functional.mse_loss(decoded[:, 1:2],   fields[:, 1:2]).item()
        mse_na_b = nn.functional.mse_loss(decoded[:, 2:12],  fields[:, 2:12]).item()
        mse_ua_b = nn.functional.mse_loss(decoded[:, 12:22], fields[:, 12:22]).item()
        loss_kl, _    = compute_cvae_kl(
            model, mu, logvar, cond,
            loss_fns["kl_weight"], loss_fns.get("free_bits", 0.0),
        )
        loss_prior_kl = compute_prior_kl(model, cond, loss_fns.get("prior_kl_weight", 0.0))

        if physics_weight > 0:
            te_pred, ti_pred, na_pred, ua_pred, \
            te_true, ti_true, na_true, ua_true = denorm_fields(decoded, fields, loss_fns["denorm"])

            energy_dict   = loss_fns["energy"].forward(
                te_pred=te_pred, ti_pred=ti_pred,
                na_pred=na_pred, ua_pred=ua_pred,
                te_true=te_true, ti_true=ti_true,
                na_true=na_true, ua_true=ua_true,
            )
            momentum_dict = loss_fns["momentum"].forward(
                na_pred=na_pred, ua_pred=ua_pred, te_pred=te_pred,
                na_true=na_true, ua_true=ua_true,
            )
            loss_energy = energy_dict["loss_total"]
            loss_momentum = momentum_dict["loss_total"]
        else:
            loss_energy = torch.tensor(0.0, device=device)
            loss_momentum = torch.tensor(0.0, device=device)

        loss_total = (
            loss_recon
            + loss_kl
            + loss_prior_kl
            + physics_weight * loss_energy
            + physics_weight * loss_momentum
        )

        totals["recon"]    += loss_recon_mse.item()   # pure MSE — used for best checkpoint
        totals["kl"]       += loss_kl.item()
        totals["prior_kl"] += loss_prior_kl.item() if isinstance(loss_prior_kl, torch.Tensor) else loss_prior_kl
        totals["energy"]   += loss_energy.item() if isinstance(loss_energy, torch.Tensor) else loss_energy
        totals["momentum"] += loss_momentum.item() if isinstance(loss_momentum, torch.Tensor) else loss_momentum
        totals["total"]    += loss_total.item()
        totals["mse_te"]   += mse_te_b
        totals["mse_ti"]   += mse_ti_b
        totals["mse_na"]   += mse_na_b
        totals["mse_ua"]   += mse_ua_b
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ConditionalVAE physics model")
    p.add_argument("--data_root",      default="a_dataset",  help="Dataset root directory")
    p.add_argument("--save_dir",       default="physics_model/checkpoints", help="Checkpoint directory")
    p.add_argument("--epochs",         type=int,   default=600)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--latent_dim",     type=int,   default=128)
    p.add_argument("--kl_weight",      type=float, default=0.01,
                   help="Target KL weight (beta) for VAE KL term after warmup")
    p.add_argument("--kl_weight_init", type=float, default=1e-3,
                   help="Initial KL weight at epoch 0 (logarithmic warmup to kl_weight). "
                        "Must be large enough to provide real gradient pressure from the start; "
                        "too small (e.g. 1e-5) causes posterior collapse before KL annealing kicks in.")
    p.add_argument("--kl_warmup_epochs", type=int, default=200,
                   help="Number of epochs to anneal KL weight from kl_weight_init to kl_weight")
    p.add_argument("--free_bits",      type=float, default=0.0,
                   help="Minimum KL (nats) per latent dimension floor. "
                        "WARNING: with latent_dim=128 and kl_weight<1, values >0.01 create an "
                        "artificial floor that blocks encoder gradients and worsens collapse. "
                        "Keep at 0.0 unless kl_weight~1.0.")
    p.add_argument("--physics_weight", type=float, default=4e-3,
                   help="Target physics weight after warmup")
    p.add_argument("--physics_weight_init", type=float, default=1e-4,
                   help="Initial physics weight at epoch 0")
    p.add_argument("--warmup_epochs",  type=int,   default=300,
                   help="Number of epochs before physics losses are fully enabled")
    p.add_argument("--ua_recon_weight", type=float, default=2.0,
                   help="Extra loss weight for ua (velocity) channels relative to te/ti/na. "
                        "ua MSE is weighted ua_recon_weight x, te/ti/na channels 1x. "
                        "Default 2.0 compensates for lower decoder capacity for bipolar fields.")
    p.add_argument("--ua_corr_weight", type=float, default=0.05,
                   help="Weight of Pearson correlation loss on ua channels. "
                        "Penalises predictions that have correct MSE but no spatial structure "
                        "(corr≈0). Loss = 1 - mean_corr(pred_ua, gt_ua). "
                        "Default 0.05 adds a modest push towards spatial pattern recovery. "
                        "Set 0.0 to disable.")
    p.add_argument("--prior_kl_weight", type=float, default=1e-3,
                   help="Weight of KL( p(z|c) || N(0,I) ) — regularises the prior network. "
                        "Keeps the learned conditional prior anchored near standard normal, "
                        "improving latent space geometry for prior sampling. "
                        "Only active when use_prior_net=True. Default 1e-3.")
    p.add_argument("--num_workers",    type=int,   default=0)
    p.add_argument("--resume",         default=None, help="Path to checkpoint to resume from")
    p.add_argument("--use_prior_net",  action="store_true", default=True,
                   help="Use prior network (cVAE mode), disable with --no_use_prior_net")
    p.add_argument("--no_use_prior_net", dest="use_prior_net", action="store_false",
                   help="Disable prior network (standard VAE mode)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info("Loading data …")
    train_handler = PlasmaDataHandler(data_root=args.data_root)
    train_handler.load_geometry()
    train_handler.load_normalization_stats()
    train_handler.load_split("train")

    val_handler = PlasmaDataHandler(data_root=args.data_root)
    val_handler.load_geometry()
    val_handler.load_normalization_stats()
    val_handler.load_split("test")

    train_ds = PlasmaDataset(train_handler)
    val_ds   = PlasmaDataset(val_handler)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    N_CHANNELS = 22  # te(1) + ti(1) + na(10) + ua(10)
    COND_DIM   = 8   # 8 input parameters X

    model = ConditionalVAE(
        in_channels=N_CHANNELS,
        out_channels=N_CHANNELS,
        latent_dim=args.latent_dim,
        cond_dim=COND_DIM,
        nx=104,
        ny=50,
        use_prior_net=args.use_prior_net,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    loss_fns = {
        "kl_weight":        args.kl_weight,   # will be overwritten each epoch by annealing
        "free_bits":        args.free_bits,
        "ua_recon_weight":  args.ua_recon_weight,
        "ua_corr_weight":   args.ua_corr_weight,
        "prior_kl_weight":  args.prior_kl_weight,
        "energy":           EnergyConservationLoss(data_handler=train_handler),
        "momentum":         MomentumConservationLoss(data_handler=train_handler),
        "denorm":           TorchDenormalizer(train_handler.norm_stats),
    }

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_recon = float("inf")  # track best by recon only — total loss changes meaning as KL weight grows

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_recon = checkpoint.get("best_val_recon", float("inf"))
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Starting training …")

    for epoch in range(start_epoch, args.epochs):
        # Compute physics weight with warmup scheduling
        physics_weight_current = compute_physics_weight_warmup(
            epoch, args.warmup_epochs, args.physics_weight_init, args.physics_weight
        )
        # KL annealing: same logarithmic warmup as physics weight.
        # Prevents posterior collapse by starting with a negligible KL penalty
        # so the encoder can first learn to use its capacity for reconstruction.
        kl_weight_current = compute_physics_weight_warmup(
            epoch, args.kl_warmup_epochs, args.kl_weight_init, args.kl_weight
        )
        loss_fns["kl_weight"] = kl_weight_current

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fns, device, physics_weight_current
        )
        val_metrics = val_epoch(
            model, val_loader, loss_fns, device, physics_weight_current
        )
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1:4d}/{args.epochs}  "
            f"train: total={train_metrics['total']:.4f} "
            f"recon={train_metrics['recon']:.4f} "
            f"kl={train_metrics['kl']:.4f} "
            f"prior_kl={train_metrics['prior_kl']:.4f} "
            f"active_dims={train_metrics['active_dims']:.0f}/{args.latent_dim} "
            f"energy={train_metrics['energy']:.4f} "
            f"momentum={train_metrics['momentum']:.4f}  |  "
            f"val: recon={val_metrics['recon']:.4f} "
            f"prior_kl={val_metrics['prior_kl']:.4f} "
            f"total={val_metrics['total']:.4f}  "
            f"kl_w={kl_weight_current:.2e}  "
            f"physics_w={physics_weight_current:.2e}  lr={lr_now:.2e}"
        )
        logger.info(
            f"  group MSE — train: te={train_metrics['mse_te']:.4f} "
            f"ti={train_metrics['mse_ti']:.4f} "
            f"na={train_metrics['mse_na']:.4f} "
            f"ua={train_metrics['mse_ua']:.4f}  |  "
            f"val: te={val_metrics['mse_te']:.4f} "
            f"ti={val_metrics['mse_ti']:.4f} "
            f"na={val_metrics['mse_na']:.4f} "
            f"ua={val_metrics['mse_ua']:.4f}"
        )

        # Save latest checkpoint
        checkpoint = {
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val_recon": best_val_recon,
            "args":           vars(args),
        }
        torch.save(checkpoint, save_dir / "latest.pt")

        # Save best checkpoint based on val reconstruction loss.
        # Total loss is NOT used for selection because it changes meaning as
        # KL weight grows during annealing — rising KL penalty would prevent
        # selecting improved reconstructions.
        if val_metrics["recon"] < best_val_recon:
            best_val_recon = val_metrics["recon"]
            torch.save(checkpoint, save_dir / "best.pt")
            logger.info(f"  -> new best val recon: {best_val_recon:.4f}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

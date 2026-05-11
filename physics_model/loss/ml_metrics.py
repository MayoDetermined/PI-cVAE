"""
KL Divergence and machine learning metrics for plasma physics models.
Includes KL divergence loss, distribution matching, and model evaluation metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KLDivergenceLoss:
    """
    KL Divergence loss for comparing probability distributions in plasma fields.
    """
    
    def __init__(self, 
                 reduction: str = 'batchmean',
                 temperature: float = 1.0,
                 epsilon: float = 1e-10):
        """
        Initialize KL Divergence loss.
        
        Args:
            reduction: 'batchmean', 'sum', 'mean', or 'none'
            temperature: Temperature for softening distributions (default 1.0)
            epsilon: Small value to avoid log(0)
        """
        self.reduction = reduction
        self.temperature = temperature
        self.epsilon = epsilon
    
    def forward(self,
                log_p: torch.Tensor,
                log_q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL(P || Q) = sum(P * (log(P) - log(Q)))
        
        This is equivalent to: KL(P || Q) = sum(P * log(P/Q))
        
        Args:
            log_p: Log probabilities of reference distribution (batch, ...)
            log_q: Log probabilities of predicted distribution (batch, ...)
        
        Returns:
            KL divergence loss
        """
        # Convert log probabilities to probabilities
        p = torch.exp(log_p)
        q = torch.exp(log_q)
        
        # Clamp to avoid numerical issues
        p = torch.clamp(p, min=self.epsilon)
        q = torch.clamp(q, min=self.epsilon)
        
        # KL divergence: sum(P * (log(P) - log(Q)))
        kl = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
        
        if self.reduction == 'batchmean':
            return torch.mean(kl)
        elif self.reduction == 'mean':
            return torch.mean(kl)
        elif self.reduction == 'sum':
            return torch.sum(kl)
        elif self.reduction == 'none':
            return kl
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def forward_unnormalized(self,
                            p: torch.Tensor,
                            q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from unnormalized distributions.
        
        Automatically normalizes inputs to form valid probability distributions.
        
        Args:
            p: Reference distribution (batch, ...) - will be normalized
            q: Predicted distribution (batch, ...) - will be normalized
        
        Returns:
            KL divergence loss
        """
        # Normalize to form probability distributions
        # Reshape to (batch, -1) for normalization
        batch_size = p.shape[0]
        p_flat = p.view(batch_size, -1)
        q_flat = q.view(batch_size, -1)
        
        # Clamp to avoid log(0)
        p_norm = torch.clamp(p_flat / p_flat.sum(dim=1, keepdim=True), min=self.epsilon)
        q_norm = torch.clamp(q_flat / q_flat.sum(dim=1, keepdim=True), min=self.epsilon)
        
        # KL divergence
        kl = torch.sum(p_norm * (torch.log(p_norm) - torch.log(q_norm)), dim=1)
        
        if self.reduction == 'batchmean' or self.reduction == 'mean':
            return torch.mean(kl)
        elif self.reduction == 'sum':
            return torch.sum(kl)
        elif self.reduction == 'none':
            return kl
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class FieldDistributionLoss:
    """
    KL divergence loss comparing probability distributions of physical fields.
    
    Converts 2D field values into probability distributions and compares them.
    Useful for ensuring predicted fields have similar statistical properties to ground truth.
    
    Incorporates geometric cell volumes for proper spatial weighting.
    """
    
    def __init__(self,
                 n_bins: int = 50,
                 data_handler = None,
                 temperature: float = 1.0,
                 use_geometry: bool = True):
        """
        Initialize field distribution loss.
        
        Args:
            n_bins: Number of bins for histogram-based distribution
            data_handler: PlasmaDataHandler for normalization and geometry info
            temperature: Temperature parameter for softening
            use_geometry: If True, weight field values by cell volumes
        """
        self.n_bins = n_bins
        self.data_handler = data_handler
        self.temperature = temperature
        self.use_geometry = use_geometry
        self.kl_loss = KLDivergenceLoss(temperature=temperature)
        
        # Compute cell volumes if geometry available
        self.cell_volumes = None
        if use_geometry and data_handler is not None:
            self._compute_cell_volumes()
    
    def _compute_cell_volumes(self) -> None:
        """
        Compute cell volumes from geometry coordinates.
        Uses corner coordinates (crx, cry) to calculate area of each cell.
        """
        if self.data_handler.crx is None or self.data_handler.cry is None:
            # Fallback: uniform grid cells
            nx, ny = self.data_handler.GRID_DIM
            self.cell_volumes = np.ones((nx, ny))
            return
        
        crx = self.data_handler.crx  # (nx, ny, 4)
        cry = self.data_handler.cry  # (nx, ny, 4)
        
        nx, ny = crx.shape[:2]
        cell_volumes = np.zeros((nx, ny))
        
        # For each cell, compute area using the Shoelace formula
        for i in range(nx):
            for j in range(ny):
                x = crx[i, j, :]
                y = cry[i, j, :]
                
                # Shoelace formula for polygon area
                area = 0.5 * np.abs(
                    np.sum(x * np.roll(y, 1)) - np.sum(y * np.roll(x, 1))
                )
                cell_volumes[i, j] = area
        
        # Normalize to unit sum for numerical stability
        self.cell_volumes = cell_volumes / np.mean(cell_volumes)
        self.cell_volumes = torch.tensor(self.cell_volumes, dtype=torch.float32)
    
    def compute_field_distribution(self,
                                   field: torch.Tensor,
                                   value_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Compute probability distribution of field values.
        
        Creates histogram of field values across spatial domain and normalizes to PDF.
        If geometry is available, weights each field value by its cell volume.
        
        Args:
            field: Physical field (batch, nx, ny) or (batch, nx, ny, n_species)
            value_range: Tuple (min, max) for histogram range. If None, uses field min/max.
        
        Returns:
            Normalized probability distribution (batch, n_bins)
        """
        batch_size = field.shape[0]
        nx, ny = field.shape[1:3]
        
        # Flatten spatial dimensions
        field_flat = field.view(batch_size, -1)  # (batch, n_points)
        
        # Prepare weights (cell volumes)
        if self.use_geometry and self.cell_volumes is not None:
            # Repeat cell volumes for each species/channel if needed
            if field.dim() == 4:
                # (batch, nx, ny, n_species) -> repeat volumes
                weights = self.cell_volumes.unsqueeze(-1).repeat(1, 1, 1, field.shape[-1])
                weights_flat = weights.view(batch_size, -1).to(field.device)
            else:
                # (batch, nx, ny) -> use volumes directly
                weights_flat = self.cell_volumes.view(1, -1).repeat(batch_size, 1).to(field.device)
            
            # Normalize weights to sum to 1 per sample
            weights_flat = weights_flat / weights_flat.sum(dim=1, keepdim=True).clamp(min=1e-10)
        else:
            # Uniform weights
            weights_flat = torch.ones_like(field_flat) / field_flat.shape[1]
        
        if value_range is None:
            # Use global min/max across all batches
            vmin = field_flat.min().detach()
            vmax = field_flat.max().detach()
        else:
            vmin, vmax = value_range
        
        # Clamp values to range
        field_clamped = torch.clamp(field_flat, min=vmin, max=vmax)
        
        # Create weighted histogram for each sample in batch
        histograms = []
        for i in range(batch_size):
            # torch.histc doesn't support weights, so use manual binning
            bin_edges = torch.linspace(float(vmin), float(vmax), self.n_bins + 1, device=field.device)
            bin_width = (vmax - vmin) / self.n_bins
            
            # Assign each value to bin
            bin_indices = torch.floor((field_clamped[i] - vmin) / bin_width).long()
            bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)
            
            # Weighted histogram
            hist = torch.zeros(self.n_bins, device=field.device)
            for b in range(self.n_bins):
                mask = bin_indices == b
                hist[b] = weights_flat[i, mask].sum()
            
            histograms.append(hist)
        
        # Stack and normalize
        hist_batch = torch.stack(histograms, dim=0)  # (batch, n_bins)
        
        # Normalize to form probability distribution
        prob_dist = hist_batch / hist_batch.sum(dim=1, keepdim=True).clamp(min=1e-10)
        
        return prob_dist
    
    def forward(self,
                field_pred: torch.Tensor,
                field_true: torch.Tensor,
                value_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Compute KL divergence between predicted and ground truth field distributions.
        
        Args:
            field_pred: Predicted field (batch, nx, ny, ...)
            field_true: Ground truth field (batch, nx, ny, ...)
            value_range: Optional value range for histogram
        
        Returns:
            KL divergence loss
        """
        # Compute distributions
        dist_true = self.compute_field_distribution(field_true, value_range)
        dist_pred = self.compute_field_distribution(field_pred, value_range)
        
        # Convert to log space for KL computation
        log_dist_true = torch.log(dist_true.clamp(min=1e-10))
        log_dist_pred = torch.log(dist_pred.clamp(min=1e-10))
        
        # Compute KL(true || pred)
        kl = torch.sum(dist_true * (log_dist_true - log_dist_pred), dim=1)
        
        return torch.mean(kl)


class SpeciesMixtureLoss:
    """
    KL divergence loss for species density distributions.
    
    Ensures that predicted mixture of species densities matches ground truth.
    Useful for maintaining correct ionization balance in plasma.
    Incorporates geometric cell volumes for proper spatial weighting.
    """
    
    def __init__(self, 
                 species_names: list = None,
                 temperature: float = 1.0,
                 data_handler = None,
                 use_geometry: bool = True):
        """
        Initialize species mixture loss.
        
        Args:
            species_names: List of species names
            temperature: Temperature for softening
            data_handler: PlasmaDataHandler for geometry info
            use_geometry: If True, weight mixture by cell volumes
        """
        self.species_names = species_names
        self.temperature = temperature
        self.data_handler = data_handler
        self.use_geometry = use_geometry
        self.kl_loss = KLDivergenceLoss(temperature=temperature)
        
        # Compute cell volumes if geometry available
        self.cell_volumes = None
        if use_geometry and data_handler is not None:
            self._compute_cell_volumes()
    
    def _compute_cell_volumes(self) -> None:
        """
        Compute cell volumes from geometry coordinates.
        """
        if self.data_handler.crx is None or self.data_handler.cry is None:
            # Fallback: uniform grid cells
            nx, ny = self.data_handler.GRID_DIM
            self.cell_volumes = np.ones((nx, ny))
            return
        
        crx = self.data_handler.crx
        cry = self.data_handler.cry
        
        nx, ny = crx.shape[:2]
        cell_volumes = np.zeros((nx, ny))
        
        for i in range(nx):
            for j in range(ny):
                x = crx[i, j, :]
                y = cry[i, j, :]
                
                area = 0.5 * np.abs(
                    np.sum(x * np.roll(y, 1)) - np.sum(y * np.roll(x, 1))
                )
                cell_volumes[i, j] = area
        
        self.cell_volumes = cell_volumes / np.mean(cell_volumes)
        self.cell_volumes = torch.tensor(self.cell_volumes, dtype=torch.float32)
    
    def compute_species_mixture(self,
                               na: torch.Tensor) -> torch.Tensor:
        """
        Compute species mixture fractions.
        
        For each spatial point, computes: f_a = n_a / sum(n_a)
        Then averages across spatial domain, weighted by cell volumes.
        
        Args:
            na: Species densities (batch, nx, ny, n_species)
        
        Returns:
            Average species mixture fractions (batch, n_species)
        """
        batch_size, nx, ny, n_species = na.shape
        
        # Total density at each point
        n_total = na.sum(dim=-1, keepdim=True).clamp(min=1e-10)  # (batch, nx, ny, 1)
        
        # Species fractions at each point
        fractions = na / n_total  # (batch, nx, ny, n_species)
        
        # Prepare weights (cell volumes)
        if self.use_geometry and self.cell_volumes is not None:
            weights = self.cell_volumes.unsqueeze(-1).to(na.device)  # (nx, ny, 1)
            weights = weights / weights.sum()  # Normalize to sum to 1
            
            # Apply weights to fractions
            weighted_fractions = fractions * weights  # (batch, nx, ny, n_species)
            avg_mixture = weighted_fractions.sum(dim=(1, 2))  # (batch, n_species)
        else:
            # Simple average over spatial domain
            avg_mixture = fractions.view(batch_size, -1, n_species).mean(dim=1)  # (batch, n_species)
        
        return avg_mixture
    
    def forward(self,
                na_pred: torch.Tensor,
                na_true: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between predicted and true species mixtures.
        
        Args:
            na_pred: Predicted species densities (batch, nx, ny, n_species)
            na_true: Ground truth species densities (batch, nx, ny, n_species)
        
        Returns:
            KL divergence loss for species mixture
        """
        # Compute mixtures
        mixture_pred = self.compute_species_mixture(na_pred)  # (batch, n_species)
        mixture_true = self.compute_species_mixture(na_true)  # (batch, n_species)
        
        # Compute KL divergence
        log_mixture_true = torch.log(mixture_true.clamp(min=1e-10))
        log_mixture_pred = torch.log(mixture_pred.clamp(min=1e-10))
        
        kl = torch.sum(mixture_true * (log_mixture_true - log_mixture_pred), dim=1)
        
        return torch.mean(kl)


class VAELatentLoss:
    """
    KL divergence loss for VAE latent space regularization.
    
    Penalizes deviation of latent distribution from standard normal N(0, I).
    Useful if using VAE for latent representation learning.
    """
    
    def __init__(self, weight: float = 0.001):
        """
        Initialize VAE latent loss.
        
        Args:
            weight: Weight for KL term (beta parameter in beta-VAE)
        """
        self.weight = weight
    
    def forward(self,
                mu: torch.Tensor,
                logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between learned latent distribution and N(0, I).
        
        KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
        
        Args:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        
        Returns:
            Weighted KL divergence loss
        """
        # KL divergence for each dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Sum over latent dimensions and average over batch
        kl_loss = kl_per_dim.sum(dim=1).mean()
        
        return self.weight * kl_loss


def compute_combined_kl_loss(predictions: Dict[str, torch.Tensor],
                            targets: Dict[str, torch.Tensor],
                            data_handler = None,
                            weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    """
    Compute combined KL divergence loss across multiple fields.
    
    Args:
        predictions: Dictionary with predicted fields
                    Keys: 'te', 'na', 'ua' (all torch.Tensor)
        targets: Dictionary with ground truth fields
        data_handler: PlasmaDataHandler instance
        weights: Dictionary with loss weights for each field
    
    Returns:
        Dictionary with individual KL losses and total
    """
    if weights is None:
        weights = {
            'te': 1.0,
            'na': 1.0,
            'ua': 0.5,
        }
    
    losses = {}
    field_losses = []
    
    # KL divergence for temperature distribution
    if 'te' in predictions and 'te' in targets:
        field_dist_loss = FieldDistributionLoss()
        kl_te = field_dist_loss.forward(predictions['te'], targets['te'])
        losses['kl_te'] = kl_te
        field_losses.append(weights.get('te', 1.0) * kl_te)
    
    # KL divergence for species mixture
    if 'na' in predictions and 'na' in targets:
        species_loss = SpeciesMixtureLoss()
        kl_na_mixture = species_loss.forward(predictions['na'], targets['na'])
        losses['kl_na_mixture'] = kl_na_mixture
        field_losses.append(weights.get('na', 1.0) * kl_na_mixture)
        
        # Also KL for density distribution
        field_dist_loss = FieldDistributionLoss()
        kl_na_dist = field_dist_loss.forward(predictions['na'], targets['na'])
        losses['kl_na_dist'] = kl_na_dist
        field_losses.append(weights.get('na', 1.0) * kl_na_dist * 0.5)
    
    # KL divergence for velocity distribution
    if 'ua' in predictions and 'ua' in targets:
        field_dist_loss = FieldDistributionLoss()
        kl_ua = field_dist_loss.forward(predictions['ua'], targets['ua'])
        losses['kl_ua'] = kl_ua
        field_losses.append(weights.get('ua', 1.0) * kl_ua)
    
    # Total loss
    total_kl = sum(field_losses) / len(field_losses) if field_losses else torch.tensor(0.0)
    losses['kl_total'] = total_kl
    
    return losses


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test KL divergence computation
    batch_size = 4
    n_bins = 50
    
    # Create sample distributions
    dist_true = torch.softmax(torch.randn(batch_size, n_bins), dim=1)
    dist_pred = torch.softmax(torch.randn(batch_size, n_bins), dim=1)
    
    # Compute KL divergence
    kl_loss = KLDivergenceLoss()
    log_true = torch.log(dist_true)
    log_pred = torch.log(dist_pred)
    
    kl = kl_loss.forward(log_true, log_pred)
    print(f"KL divergence: {kl:.4f}")
    
    # Test field distribution loss
    field_pred = torch.randn(batch_size, 104, 50)
    field_true = torch.randn(batch_size, 104, 50)
    
    field_dist_loss = FieldDistributionLoss(n_bins=50)
    kl_field = field_dist_loss.forward(field_pred, field_true)
    print(f"Field distribution KL: {kl_field:.4f}")
    
    # Test species mixture loss
    na_pred = torch.rand(batch_size, 104, 50, 10)
    na_true = torch.rand(batch_size, 104, 50, 10)
    
    species_loss = SpeciesMixtureLoss()
    kl_species = species_loss.forward(na_pred, na_true)
    print(f"Species mixture KL: {kl_species:.4f}")
    
    # Test VAE latent loss
    mu = torch.randn(batch_size, 64)
    logvar = torch.randn(batch_size, 64)
    
    vae_loss = VAELatentLoss(weight=0.001)
    kl_vae = vae_loss.forward(mu, logvar)
    print(f"VAE latent KL: {kl_vae:.4f}")

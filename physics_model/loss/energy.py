"""
Energy conservation loss function for plasma physics models.
Incorporates thermal (potential) and kinetic energy with geometric considerations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# Atomic masses (in atomic mass units, u)
# Used for kinetic energy calculation
ATOMIC_MASSES = {
    'D0': 2.0,      # Deuterium neutral
    'D1': 2.0,      # Deuterium ion (same mass)
    'N0': 14.0,     # Nitrogen neutral
    'N1': 14.0,     # Nitrogen ion N+1
    'N2': 14.0,     # Nitrogen ion N+2
    'N3': 14.0,     # Nitrogen ion N+3
    'N4': 14.0,     # Nitrogen ion N+4
    'N5': 14.0,     # Nitrogen ion N+5
    'N6': 14.0,     # Nitrogen ion N+6
    'N7': 14.0,     # Nitrogen ion N+7
}

# Convert atomic mass units to kg (1 u = 1.66053906660e-27 kg)
AMU_TO_KG = 1.66053906660e-27


class EnergyConservationLoss:
    """
    Loss function for energy conservation in plasma simulations.
    
    Computes energy conservation penalty based on:
    - Thermal (potential) energy from temperatures and densities
    - Kinetic energy from particle velocities
    - Geometric cell volumes/areas
    - Poloidal circularity (periodic boundary conditions along poloidal axis)
    """
    
    def __init__(self, 
                 data_handler,
                 species_names: list = None,
                 weight_thermal: float = 1.0,
                 weight_kinetic: float = 1.0,
                 weight_flux: float = 0.1,
                 weight_poloidal_periodicity: float = 0.05):
        """
        Initialize energy conservation loss.
        
        Args:
            data_handler: PlasmaDataHandler instance with geometry and normalization info
            species_names: List of species names (from data_handler.SPECIES)
            weight_thermal: Weight for thermal energy term
            weight_kinetic: Weight for kinetic energy term
            weight_flux: Weight for energy flux conservation term
            weight_poloidal_periodicity: Weight for poloidal circularity constraint
        """
        self.data_handler = data_handler
        self.species_names = species_names or data_handler.SPECIES
        self.weight_thermal = weight_thermal
        self.weight_kinetic = weight_kinetic
        self.weight_flux = weight_flux
        self.weight_poloidal_periodicity = weight_poloidal_periodicity
        
        # Prepare mass array for kinetic energy calculation
        self.masses = np.array([ATOMIC_MASSES[name] for name in self.species_names])
        self.masses_kg = torch.tensor(self.masses * AMU_TO_KG, dtype=torch.float32)
        
        # Compute cell volumes/areas from geometry
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
        # Corners are ordered: [0]=LL, [1]=LR, [2]=UR, [3]=UL
        for i in range(nx):
            for j in range(ny):
                x = crx[i, j, :]  # x-coordinates of 4 corners
                y = cry[i, j, :]  # y-coordinates of 4 corners
                
                # Shoelace formula for polygon area
                area = 0.5 * np.abs(
                    np.sum(x * np.roll(y, 1)) - np.sum(y * np.roll(x, 1))
                )
                cell_volumes[i, j] = area
        
        # Normalize to unit sum for numerical stability
        self.cell_volumes = cell_volumes / np.mean(cell_volumes)
        self.cell_volumes = torch.tensor(self.cell_volumes, dtype=torch.float32)
    
    def compute_thermal_energy(self,
                              te: torch.Tensor,
                              ti: torch.Tensor,
                              na: torch.Tensor) -> torch.Tensor:
        """
        Compute thermal (potential) energy density.
        
        E_thermal = integral over domain of sum_species(n_a * (T_e + T_i))
        
        Args:
            te: Electron temperature (batch, nx, ny) or (batch, nx, ny, 1)
            ti: Ion temperature (batch, nx, ny) or (batch, nx, ny, 1)
            na: Species densities (batch, nx, ny, n_species)
        
        Returns:
            Thermal energy per sample (batch,)
        """
        # Handle channel dimension if present
        if te.dim() == 4:
            te = te.squeeze(-1)
        if ti.dim() == 4:
            ti = ti.squeeze(-1)
        
        # Ensure te and ti are broadcast-compatible with na
        batch_size = na.shape[0]
        nx, ny = na.shape[1:3]
        
        # Expand to (batch, nx, ny, n_species)
        te_expanded = te.unsqueeze(-1).expand(-1, -1, -1, na.shape[-1])
        ti_expanded = ti.unsqueeze(-1).expand(-1, -1, -1, na.shape[-1])
        
        # Thermal energy density: n_a * (T_e + T_i) [eV]
        # Note: temperatures are in eV, densities in m^-3
        energy_density = na * (te_expanded + ti_expanded)  # (batch, nx, ny, n_species)
        
        # Sum over species and multiply by cell volumes
        cell_volumes = self.cell_volumes.to(energy_density.device)
        if cell_volumes.dim() == 2:
            cell_volumes = cell_volumes.unsqueeze(0).unsqueeze(-1)
        
        thermal_energy = (energy_density * cell_volumes).sum(dim=(1, 2, 3))  # (batch,)
        
        return thermal_energy
    
    def compute_kinetic_energy(self,
                              na: torch.Tensor,
                              ua: torch.Tensor) -> torch.Tensor:
        """
        Compute kinetic energy density.
        
        E_kinetic = integral over domain of 0.5 * sum_species(m_a * n_a * u_a^2)
        
        Args:
            na: Species densities (batch, nx, ny, n_species)
            ua: Species velocities (batch, nx, ny, n_species)
        
        Returns:
            Kinetic energy per sample (batch,)
        """
        # Prepare mass tensor
        masses = self.masses_kg.to(ua.device)
        if masses.dim() == 1:
            masses = masses.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Kinetic energy density: 0.5 * m_a * n_a * u_a^2 [Joules per m^3]
        kinetic_density = 0.5 * masses * na * (ua ** 2)  # (batch, nx, ny, n_species)
        
        # Sum over species and multiply by cell volumes
        cell_volumes = self.cell_volumes.to(kinetic_density.device)
        if cell_volumes.dim() == 2:
            cell_volumes = cell_volumes.unsqueeze(0).unsqueeze(-1)
        
        kinetic_energy = (kinetic_density * cell_volumes).sum(dim=(1, 2, 3))  # (batch,)
        
        return kinetic_energy
    
    def compute_energy_flux_boundary(self,
                                     te: torch.Tensor,
                                     ti: torch.Tensor,
                                     na: torch.Tensor,
                                     ua: torch.Tensor) -> torch.Tensor:
        """
        Compute energy flux at domain boundary (for open domain simulation).
        Models energy leaving through boundaries.
        
        Args:
            te: Electron temperature (batch, nx, ny)
            ti: Ion temperature (batch, nx, ny)
            na: Species densities (batch, nx, ny, n_species)
            ua: Species velocities (batch, nx, ny, n_species)
        
        Returns:
            Energy flux estimate (batch,)
        """
        # Handle channel dimension if present
        if te.dim() == 4:
            te = te.squeeze(-1)
        if ti.dim() == 4:
            ti = ti.squeeze(-1)
        
        # Extract boundary values (e.g., last row and column)
        # Simplified: use mean boundary temperature and density
        te_boundary_x = te[:, -1, :].mean(dim=-1)    # (batch,)
        te_boundary_y = te[:, :, -1].mean(dim=-1)    # (batch,)
        te_boundary = (te_boundary_x + te_boundary_y) / 2
        
        na_boundary_x = na[:, -1, :, :].mean(dim=(1, 2))    # mean over ny, n_species → (batch,)
        na_boundary_y = na[:, :, -1, :].mean(dim=(1, 2))    # mean over nx, n_species → (batch,)
        na_boundary = (na_boundary_x + na_boundary_y) / 2
        
        ua_boundary_x = ua[:, -1, :, :].mean(dim=(1, 2))    # mean over ny, n_species → (batch,)
        ua_boundary_y = ua[:, :, -1, :].mean(dim=(1, 2))    # mean over nx, n_species → (batch,)
        ua_boundary = (ua_boundary_x + ua_boundary_y) / 2
        
        # Energy flux ~ n * T * u (simplified model)
        energy_flux = (na_boundary * te_boundary * ua_boundary).mean()
        
        return energy_flux
    
    def compute_poloidal_flux(self,
                             te: torch.Tensor,
                             ti: torch.Tensor,
                             na: torch.Tensor,
                             ua: torch.Tensor) -> torch.Tensor:
        """
        Compute energy flux in poloidal direction (along y-axis).
        Takes into account the circularity/periodicity of poloidal axis.
        
        In tokamaks, poloidal direction is periodic (circular).
        Energy flux: Phi = integral(n * T * u * dS)
        
        Args:
            te: Electron temperature (batch, nx, ny)
            ti: Ion temperature (batch, nx, ny)
            na: Species densities (batch, nx, ny, n_species)
            ua: Species velocities (batch, nx, ny, n_species)
        
        Returns:
            Poloidal energy flux for each toroidal position (batch, nx)
        """
        if te.dim() == 4:
            te = te.squeeze(-1)
        if ti.dim() == 4:
            ti = ti.squeeze(-1)
        
        batch_size, nx, ny = te.shape
        n_species = na.shape[-1]
        
        # Combine temperatures: total thermal energy per particle
        temp_total = te + ti if ti is not None else te  # (batch, nx, ny)
        
        # Expand to species dimension
        temp_expanded = temp_total.unsqueeze(-1).expand(-1, -1, -1, n_species)  # (batch, nx, ny, n_species)
        
        # Energy flux density: n_a * T * u_a
        flux_density = na * temp_expanded * ua  # (batch, nx, ny, n_species)
        
        # Sum over species and integrate along poloidal direction
        poloidal_flux = flux_density.sum(dim=-1).sum(dim=-1)  # (batch, nx)
        
        return poloidal_flux
    
    def check_poloidal_periodicity(self,
                                  field: torch.Tensor,
                                  method: str = 'mse') -> torch.Tensor:
        """
        Check periodicity at poloidal boundaries (y-direction).
        Enforces continuity between last and first poloidal cells.
        
        In circular geometry, field[..., -1, :] should be close to field[..., 0, :]
        
        Args:
            field: Physical field (batch, nx, ny, ...) or (batch, nx, ny)
            method: 'mse' (L2 distance) or 'grad' (gradient mismatch)
        
        Returns:
            Periodicity violation penalty (scalar)
        """
        # Poloidal direction is y (spatial dim 2), explicit indexing avoids the
        # ambiguity of field[..., k, :] which selects dim -2, not dim 2 for 3D tensors.
        boundary_start = field[:, :, 0]    # (B, nx) or (B, nx, n_spe) - first poloidal
        boundary_end   = field[:, :, -1]   # (B, nx) or (B, nx, n_spe) - last poloidal
        
        # Relative scale: prevents O(1e42) MSE when field magnitudes are large
        # (e.g., na ~1e21 m^-3 would give raw MSE ~1e42 for 1% boundary mismatch).
        scale = field.detach().abs().mean().clamp(min=1e-30)
        
        if method == 'mse':
            # Relative L2 distance between boundaries (dimensionless)
            periodicity_loss = F.mse_loss(boundary_end / scale, boundary_start / scale)
        elif method == 'grad':
            # Relative gradient mismatch along poloidal axis
            grad_start_forward = field[:, :, 1]  - field[:, :, 0]   # (B, nx[, n_spe])
            grad_end_wrapped   = field[:, :, 0]  - field[:, :, -1]  # wrap: last->first
            
            periodicity_loss = F.mse_loss(grad_start_forward / scale, grad_end_wrapped / scale)
        else:
            raise ValueError(f"Unknown periodicity method: {method}")
        
        return periodicity_loss
    
    def forward(self,
                te_pred: torch.Tensor,
                ti_pred: torch.Tensor,
                na_pred: torch.Tensor,
                ua_pred: torch.Tensor,
                te_true: Optional[torch.Tensor] = None,
                ti_true: Optional[torch.Tensor] = None,
                na_true: Optional[torch.Tensor] = None,
                ua_true: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute energy conservation loss with poloidal circularity constraint.
        
        Two modes:
        1. If true values provided: compare predicted vs true energy
        2. If true values not provided: enforce energy stability (low variation)
        
        Also enforces poloidal periodicity for all fields.
        
        Args:
            te_pred, ti_pred, na_pred, ua_pred: Predicted fields
            te_true, ti_true, na_true, ua_true: Ground truth fields (optional)
        
        Returns:
            Dictionary with loss components:
            - 'loss_total': Total energy conservation loss
            - 'loss_thermal': Thermal energy matching
            - 'loss_kinetic': Kinetic energy matching
            - 'loss_flux': Energy flux penalty
            - 'loss_poloidal_periodicity': Poloidal circularity constraint
        """
        # Compute thermal energy
        thermal_pred = self.compute_thermal_energy(te_pred, ti_pred, na_pred)
        kinetic_pred = self.compute_kinetic_energy(na_pred, ua_pred)
        total_energy_pred = thermal_pred + kinetic_pred
        
        losses = {}
        
        if te_true is not None and na_true is not None:
            # Compare with ground truth
            thermal_true = self.compute_thermal_energy(te_true, ti_true, na_true)
            kinetic_true = self.compute_kinetic_energy(na_true, ua_true)
            total_energy_true = thermal_true + kinetic_true
            
            # Relative MSE: divide by true energy scale to avoid float32 overflow
            # (thermal energy O(1e28) would cause MSE ≈ 1e56 → inf in float32)
            thermal_scale = thermal_true.abs().mean().clamp(min=1.0).detach()
            kinetic_scale = kinetic_true.abs().mean().clamp(min=1.0).detach()
            loss_thermal = F.mse_loss(thermal_pred / thermal_scale, thermal_true / thermal_scale)
            loss_kinetic = F.mse_loss(kinetic_pred / kinetic_scale, kinetic_true / kinetic_scale)
            
            losses['loss_thermal'] = loss_thermal
            losses['loss_kinetic'] = loss_kinetic
        else:
            # Enforce energy stability (minimize variance across spatial domain)
            # Lower variance = more stable/conserved energy
            loss_thermal = torch.var(thermal_pred)
            loss_kinetic = torch.var(kinetic_pred)
            
            losses['loss_thermal'] = loss_thermal * 0.1
            losses['loss_kinetic'] = loss_kinetic * 0.1
        
        # Energy flux: relative MSE to handle large physical magnitudes
        energy_flux_pred = self.compute_energy_flux_boundary(te_pred, ti_pred, na_pred, ua_pred)
        if te_true is not None and na_true is not None:
            energy_flux_true = self.compute_energy_flux_boundary(te_true, ti_true, na_true, ua_true)
            flux_scale = energy_flux_true.detach().abs().clamp(min=1e-30)
            loss_flux = ((energy_flux_pred - energy_flux_true.detach()) / flux_scale).pow(2)
        else:
            loss_flux = energy_flux_pred.abs()
        
        losses['loss_flux'] = loss_flux
        
        # ============== POLOIDAL CIRCULARITY CONSTRAINTS ==============
        # Enforce periodicity along poloidal axis (y-direction)
        # All physical fields must satisfy periodic boundary conditions
        
        loss_poloidal_te = self.check_poloidal_periodicity(te_pred, method='mse')
        loss_poloidal_ti = self.check_poloidal_periodicity(ti_pred, method='mse') if ti_pred is not None else torch.tensor(0.0, device=te_pred.device)
        loss_poloidal_na = self.check_poloidal_periodicity(na_pred, method='mse')
        loss_poloidal_ua = self.check_poloidal_periodicity(ua_pred, method='mse')
        
        # Combined poloidal periodicity loss
        loss_poloidal_periodicity = (
            loss_poloidal_te + 
            (loss_poloidal_ti if isinstance(loss_poloidal_ti, torch.Tensor) else 0.0) + 
            loss_poloidal_na + 
            loss_poloidal_ua
        ) / 4.0
        
        losses['loss_poloidal_periodicity'] = loss_poloidal_periodicity
        
        # Compute total loss with weights
        loss_total = (
            self.weight_thermal * losses['loss_thermal'] +
            self.weight_kinetic * losses['loss_kinetic'] +
            self.weight_flux * loss_flux +
            self.weight_poloidal_periodicity * loss_poloidal_periodicity
        )
        
        losses['loss_total'] = loss_total
        
        return losses


def energy_conservation_loss(predictions: Dict[str, torch.Tensor],
                            targets: Optional[Dict[str, torch.Tensor]] = None,
                            data_handler = None,
                            **kwargs) -> torch.Tensor:
    """
    Convenience function for energy conservation loss calculation.
    
    Args:
        predictions: Dictionary with predicted fields
                    Keys: 'te', 'ti', 'na', 'ua' (all torch.Tensor)
        targets: Optional dictionary with ground truth fields
        data_handler: PlasmaDataHandler instance
        **kwargs: Additional arguments passed to EnergyConservationLoss
    
    Returns:
        Total energy conservation loss
    """
    if data_handler is None:
        raise ValueError("data_handler must be provided")
    
    loss_fn = EnergyConservationLoss(data_handler, **kwargs)
    
    target_dict = targets or {}
    
    loss_dict = loss_fn.forward(
        te_pred=predictions['te'],
        ti_pred=predictions['ti'],
        na_pred=predictions['na'],
        ua_pred=predictions['ua'],
        te_true=target_dict.get('te'),
        ti_true=target_dict.get('ti'),
        na_true=target_dict.get('na'),
        ua_true=target_dict.get('ua'),
    )
    
    return loss_dict['loss_total']

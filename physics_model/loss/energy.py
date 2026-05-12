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

# Charge numbers (ionization state) per species
# Used for quasineutrality: n_e = sum_a Z_a * n_a
CHARGE_NUMBERS = {
    'D0': 0,    # Deuterium neutral
    'D1': 1,    # Deuterium ion
    'N0': 0,    # Nitrogen neutral
    'N1': 1,    # N+1
    'N2': 2,    # N+2
    'N3': 3,    # N+3
    'N4': 4,    # N+4
    'N5': 5,    # N+5
    'N6': 6,    # N+6
    'N7': 7,    # N+7
}

# Convert atomic mass units to kg (1 u = 1.66053906660e-27 kg)
AMU_TO_KG = 1.66053906660e-27

# Convert eV to Joules (elementary charge)
EV_TO_J = 1.602176634e-19  # J/eV

# Thermal degrees of freedom factor f/2 (f=3 for 3D ideal plasma)
THERMAL_DOF_FACTOR = 1.5


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

        # Charge numbers for quasineutrality: n_e = sum_a Z_a * n_a
        charges = np.array([CHARGE_NUMBERS[name] for name in self.species_names])
        self.charges_t = torch.tensor(charges, dtype=torch.float32)
        
        # Compute cell volumes/areas from geometry
        self._compute_cell_volumes()
    
    def _compute_cell_volumes(self) -> None:
        """
        Compute cell volumes from geometry coordinates.
        Uses corner coordinates (crx, cry) to calculate area of each cell.
        """
        if self.data_handler.crx is None or self.data_handler.cry is None:
            # Fallback: uniform grid cells and unit face lengths
            nx, ny = self.data_handler.GRID_DIM
            self.cell_volumes    = torch.ones((nx, ny), dtype=torch.float32)
            self.face_len_outer    = torch.ones(ny,     dtype=torch.float32)
            self.face_len_pol_low  = torch.ones(nx,     dtype=torch.float32)
            self.face_len_pol_high = torch.ones(nx,     dtype=torch.float32)
            return

        crx = self.data_handler.crx  # (nx, ny, 4)  — R [m]
        cry = self.data_handler.cry  # (nx, ny, 4)  — Z [m]

        nx, ny = crx.shape[:2]
        cell_volumes = np.zeros((nx, ny))

        # Shoelace formula: 2D cell area in the (R, Z) poloidal plane [m^2]
        # Vectorised over all cells
        x = crx  # (nx, ny, 4)
        y = cry
        # Shoelace: A = 0.5 * |sum_k x_k*(y_{k-1} - y_{k+1})|
        cell_volumes = 0.5 * np.abs(
            x[:, :, 0] * (y[:, :, 3] - y[:, :, 1]) +
            x[:, :, 1] * (y[:, :, 0] - y[:, :, 2]) +
            x[:, :, 2] * (y[:, :, 1] - y[:, :, 3]) +
            x[:, :, 3] * (y[:, :, 2] - y[:, :, 0])
        )

        # Toroidal Jacobian: in cylindrical (R, phi, Z) geometry the 3-D volume
        # element is  dV = R * dR * dZ * dphi = 2*pi*R * dA  (assuming toroidal
        # symmetry, i.e. the SOLPS 2-D approximation).  crx stores R [m], so
        # R_center = mean of the 4 corner R-values for each cell.
        R_center = crx.mean(axis=2)  # (nx, ny)
        cell_volumes = cell_volumes * R_center  # weight by toroidal Jacobian

        # Normalise to unit mean for numerical stability
        mean_vol = cell_volumes[cell_volumes > 0].mean()
        self.cell_volumes = torch.tensor(cell_volumes / mean_vol, dtype=torch.float32)

        # Boundary face lengths (same SOLPS corner convention as MomentumConservationLoss)
        def _face_len(x0, y0, x1, y1):
            return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Outer radial face (i = nx-1): corners 1 → 2  → shape (ny,)
        fl_outer = _face_len(crx[-1, :, 1], cry[-1, :, 1], crx[-1, :, 2], cry[-1, :, 2])
        self.face_len_outer = torch.tensor(fl_outer / np.mean(fl_outer), dtype=torch.float32)

        # Lower poloidal target (j = 0): corners 0 → 1  → shape (nx,)
        fl_pol_low = _face_len(crx[:, 0, 0], cry[:, 0, 0], crx[:, 0, 1], cry[:, 0, 1])
        self.face_len_pol_low = torch.tensor(fl_pol_low / np.mean(fl_pol_low), dtype=torch.float32)

        # Upper poloidal target (j = ny-1): corners 3 → 2  → shape (nx,)
        fl_pol_high = _face_len(crx[:, -1, 3], cry[:, -1, 3], crx[:, -1, 2], cry[:, -1, 2])
        self.face_len_pol_high = torch.tensor(fl_pol_high / np.mean(fl_pol_high), dtype=torch.float32)
    
    def compute_thermal_energy(self,
                              te: torch.Tensor,
                              ti: torch.Tensor,
                              na: torch.Tensor) -> torch.Tensor:
        """
        Compute thermal energy integrated over the domain.

        Correct physical split (quasineutral plasma):
          n_e = sum_a Z_a * n_a          (quasineutrality)
          E_e = (3/2) * n_e * e * T_e   [J/m^3]
          E_i = (3/2) * sum_a n_a * e * T_i  [J/m^3]  (shared T_i)
          E_thermal = integral (E_e + E_i) dV

        Args:
            te: Electron temperature [eV] (batch, nx, ny) or (batch, nx, ny, 1)
            ti: Ion temperature [eV]      (batch, nx, ny) or (batch, nx, ny, 1)
            na: Species densities [m^-3] (batch, nx, ny, n_species)

        Returns:
            Thermal energy per sample [J] (batch,)
        """
        if te.dim() == 4:
            te = te.squeeze(-1)
        if ti.dim() == 4:
            ti = ti.squeeze(-1)

        cell_volumes = self.cell_volumes.to(na.device)  # (nx, ny)
        charges = self.charges_t.to(na.device)          # (n_species,)

        # Electron density from quasineutrality: n_e = sum_a Z_a * n_a
        n_e = (na * charges).sum(dim=-1)  # (batch, nx, ny)

        # Electron thermal energy density: (3/2) * n_e * e * T_e  [J/m^3]
        e_density_electrons = THERMAL_DOF_FACTOR * EV_TO_J * n_e * te  # (batch, nx, ny)

        # Ion thermal energy density: (3/2) * (sum_a n_a) * e * T_i  [J/m^3]
        n_ions_total = na.sum(dim=-1)  # (batch, nx, ny)
        e_density_ions = THERMAL_DOF_FACTOR * EV_TO_J * n_ions_total * ti  # (batch, nx, ny)

        # Integrate over cell volumes
        e_density_total = e_density_electrons + e_density_ions  # (batch, nx, ny)
        thermal_energy = (e_density_total * cell_volumes.unsqueeze(0)).sum(dim=(1, 2))  # (batch,)

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
        Compute energy flux at domain boundary.

        Integrates  n_e * T_e * |u_mean|  pointwise over the three outflow
        surfaces (outer radial wall, lower and upper poloidal targets), weighted
        by normalised face lengths.  Using |u| instead of u avoids cancellation
        from bipolar parallel velocities.

        Args:
            te: Electron temperature (batch, nx, ny)
            ti: Ion temperature (batch, nx, ny)  [unused, kept for signature]
            na: Species densities (batch, nx, ny, n_species)
            ua: Species velocities (batch, nx, ny, n_species)

        Returns:
            Energy flux per sample (batch,)
        """
        if te.dim() == 4:
            te = te.squeeze(-1)

        charges = self.charges_t.to(na.device)        # (n_species,)
        # Electron density from quasineutrality: n_e = sum_a Z_a * n_a
        n_e = (na * charges).sum(dim=-1)              # (batch, nx, ny)

        # Mean |u| across species to avoid bipolar cancellation
        ua_abs = ua.abs().mean(dim=-1)                # (batch, nx, ny)

        # Pointwise energy flux density: n_e * T_e * |u|  (batch, nx, ny)
        flux = n_e * te * ua_abs

        # Integrate over the three outflow boundary faces
        face_outer    = self.face_len_outer.to(ua.device)     # (ny,)
        face_pol_low  = self.face_len_pol_low.to(ua.device)   # (nx,)
        face_pol_high = self.face_len_pol_high.to(ua.device)  # (nx,)

        # Outer radial wall (i = nx-1): sum over ny
        flux_outer    = (flux[:, -1, :]  * face_outer.unsqueeze(0)).sum(dim=-1)    # (batch,)
        # Lower poloidal target (j = 0): sum over nx
        flux_pol_low  = (flux[:, :,  0]  * face_pol_low.unsqueeze(0)).sum(dim=-1)  # (batch,)
        # Upper poloidal target (j = ny-1): sum over nx
        flux_pol_high = (flux[:, :, -1]  * face_pol_high.unsqueeze(0)).sum(dim=-1) # (batch,)

        return flux_outer + flux_pol_low + flux_pol_high
    
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
            # Scale by sum of both magnitudes: parallel velocities at the boundary are
            # bidirectional in tokamak geometry, so the signed mean (energy_flux_true)
            # can be near zero even when physical fluxes are large.  Using only
            # |energy_flux_true| as the denominator with a tiny clamp floor (1e-30)
            # causes division by ~0 → Inf loss → NaN gradients after clip_grad_norm_.
            flux_scale = (energy_flux_pred.detach().abs() + energy_flux_true.detach().abs()).clamp(min=1.0)
            loss_flux = ((energy_flux_pred - energy_flux_true.detach()) / flux_scale).pow(2).mean()
        else:
            loss_flux = energy_flux_pred.abs().mean()
        
        losses['loss_flux'] = loss_flux
        
        # Poloidal periodicity is intentionally disabled:
        # In SOLPS divertor geometry the j-axis is RADIAL (core → wall), so
        # j=0 (core/separatrix) and j=ny-1 (outer wall) have completely different
        # plasma conditions and are NOT periodic.  Enforcing field[:,j=0] ≈
        # field[:,j=-1] creates a permanent, non-minimisable penalty that injects
        # false gradient signal and can hurt convergence of ua near boundaries.
        loss_poloidal_periodicity = torch.tensor(0.0, device=te_pred.device)
        losses['loss_poloidal_periodicity'] = loss_poloidal_periodicity

        # Compute total loss with weights
        loss_total = (
            self.weight_thermal * losses['loss_thermal'] +
            self.weight_kinetic * losses['loss_kinetic'] +
            self.weight_flux * loss_flux
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

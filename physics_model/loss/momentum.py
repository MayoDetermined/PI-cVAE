"""
Momentum conservation loss function for plasma physics models.
Incorporates momentum balance with geometric considerations and poloidal circularity.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# Atomic masses (in atomic mass units, u)
ATOMIC_MASSES = {
    'D0': 2.0,      # Deuterium neutral
    'D1': 2.0,      # Deuterium ion
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

# Elementary charge
ELEMENTARY_CHARGE = 1.60217663e-19


class MomentumConservationLoss:
    """
    Loss function for momentum conservation in plasma simulations.
    
    Computes momentum conservation penalty based on:
    - Particle momentum (m * n * u) from all species
    - Electromagnetic force effects (magnetic confinement)
    - Geometric cell volumes
    - Poloidal circularity (periodic boundary conditions)
    - Momentum flux and exchange at boundaries
    """
    
    def __init__(self, 
                 data_handler,
                 species_names: list = None,
                 weight_momentum: float = 1.0,
                 weight_momentum_flux: float = 0.5,
                 weight_poloidal_periodicity: float = 0.05,
                 weight_charge_effects: float = 0.1):
        """
        Initialize momentum conservation loss.
        
        Args:
            data_handler: PlasmaDataHandler instance with geometry and normalization info
            species_names: List of species names (from data_handler.SPECIES)
            weight_momentum: Weight for total momentum balance term
            weight_momentum_flux: Weight for momentum flux conservation
            weight_poloidal_periodicity: Weight for poloidal circularity constraint
            weight_charge_effects: Weight for charge-dependent effects (ions vs neutrals)
        """
        self.data_handler = data_handler
        self.species_names = species_names or data_handler.SPECIES
        self.weight_momentum = weight_momentum
        self.weight_momentum_flux = weight_momentum_flux
        self.weight_poloidal_periodicity = weight_poloidal_periodicity
        self.weight_charge_effects = weight_charge_effects
        
        # Prepare mass array
        self.masses = np.array([ATOMIC_MASSES[name] for name in self.species_names])
        self.masses_kg = torch.tensor(self.masses * AMU_TO_KG, dtype=torch.float32)
        
        # Determine ionization state for each species
        # Species naming: D0 (neutral), D1 (ion), N0 (neutral), N1-N7 (ions)
        self.charges = self._compute_charges()
        
        # Compute cell volumes/areas from geometry
        self._compute_cell_volumes()
    
    def _compute_charges(self) -> np.ndarray:
        """
        Determine charge state for each species.
        
        Returns:
            Array of charge states (0 for neutrals, Z+ for ions)
        """
        charges = np.zeros(len(self.species_names))
        
        for i, name in enumerate(self.species_names):
            if 'D0' in name or 'N0' in name:
                charges[i] = 0  # Neutral
            elif 'D1' in name or 'N1' in name:
                charges[i] = 1  # +1 ion
            elif 'N2' in name:
                charges[i] = 2  # +2 ion
            elif 'N3' in name:
                charges[i] = 3  # +3 ion
            elif 'N4' in name:
                charges[i] = 4  # +4 ion
            elif 'N5' in name:
                charges[i] = 5  # +5 ion
            elif 'N6' in name:
                charges[i] = 6  # +6 ion
            elif 'N7' in name:
                charges[i] = 7  # +7 ion
        
        return charges
    
    def _compute_cell_volumes(self) -> None:
        """
        Compute cell volumes and boundary face lengths from geometry coordinates.
        Uses corner coordinates (crx, cry) to calculate the area of each cell
        and the lengths of the three outflow boundary faces.

        SOLPS-ITER corner convention (0-indexed):
          0 = inner-lower  (small i, small j)
          1 = outer-lower  (large i, small j)
          2 = outer-upper  (large i, large j)
          3 = inner-upper  (small i, large j)

        Derived boundary faces:
          Outer radial face  (i = nx-1): corners 1 → 2
          Lower poloidal face (j = 0)  : corners 0 → 1
          Upper poloidal face (j = ny-1): corners 3 → 2
        """
        if self.data_handler.crx is None or self.data_handler.cry is None:
            # Fallback: uniform grid cells and unit face lengths
            nx, ny = self.data_handler.GRID_DIM
            self.cell_volumes    = torch.ones((nx, ny),  dtype=torch.float32)
            self.face_len_outer    = torch.ones(ny,       dtype=torch.float32)
            self.face_len_pol_low  = torch.ones(nx,       dtype=torch.float32)
            self.face_len_pol_high = torch.ones(nx,       dtype=torch.float32)
            return

        crx = self.data_handler.crx  # (nx, ny, 4)
        cry = self.data_handler.cry  # (nx, ny, 4)

        nx, ny = crx.shape[:2]
        cell_volumes = np.zeros((nx, ny))

        # Cell areas via the Shoelace formula
        for i in range(nx):
            for j in range(ny):
                x = crx[i, j, :]
                y = cry[i, j, :]
                cell_volumes[i, j] = 0.5 * np.abs(
                    np.sum(x * np.roll(y, 1)) - np.sum(y * np.roll(x, 1))
                )

        # Normalize to unit mean for numerical stability
        self.cell_volumes = torch.tensor(
            cell_volumes / np.mean(cell_volumes), dtype=torch.float32
        )

        def _face_len(x0, y0, x1, y1):
            """Euclidean length of a face between two corner arrays."""
            return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Outer radial face (i = nx-1): between corners 1 and 2  →  shape (ny,)
        fl_outer = _face_len(
            crx[-1, :, 1], cry[-1, :, 1],
            crx[-1, :, 2], cry[-1, :, 2],
        )
        self.face_len_outer = torch.tensor(
            fl_outer / np.mean(fl_outer), dtype=torch.float32
        )

        # Lower poloidal face (j = 0): between corners 0 and 1  →  shape (nx,)
        fl_pol_low = _face_len(
            crx[:, 0, 0], cry[:, 0, 0],
            crx[:, 0, 1], cry[:, 0, 1],
        )
        self.face_len_pol_low = torch.tensor(
            fl_pol_low / np.mean(fl_pol_low), dtype=torch.float32
        )

        # Upper poloidal face (j = ny-1): between corners 3 and 2  →  shape (nx,)
        fl_pol_high = _face_len(
            crx[:, -1, 3], cry[:, -1, 3],
            crx[:, -1, 2], cry[:, -1, 2],
        )
        self.face_len_pol_high = torch.tensor(
            fl_pol_high / np.mean(fl_pol_high), dtype=torch.float32
        )
    
    def compute_total_momentum(self,
                              na: torch.Tensor,
                              ua: torch.Tensor) -> torch.Tensor:
        """
        Compute total momentum density integrated over domain.
        
        p_total = integral over domain of sum_species(m_a * n_a * u_a)
        
        Args:
            na: Species densities (batch, nx, ny, n_species)
            ua: Species velocities (batch, nx, ny, n_species)
        
        Returns:
            Total momentum per sample (batch,)
        """
        # Prepare mass tensor
        masses = self.masses_kg.to(ua.device)
        if masses.dim() == 1:
            masses = masses.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Momentum density: m_a * n_a * u_a (kg/m^2 * s^-1)
        momentum_density = masses * na * ua  # (batch, nx, ny, n_species)
        
        # Sum over species and multiply by cell volumes
        cell_volumes = self.cell_volumes.to(momentum_density.device)
        if cell_volumes.dim() == 2:
            cell_volumes = cell_volumes.unsqueeze(0).unsqueeze(-1)
        
        total_momentum = (momentum_density * cell_volumes).sum(dim=(1, 2, 3))  # (batch,)
        
        return total_momentum
    
    def compute_charge_effects(self,
                              na: torch.Tensor) -> torch.Tensor:
        """
        Compute quasi-neutrality penalty.

        Enforces that the net charge density rho_q = sum_a(Z_a * n_a) is
        spatially uniform (low variance), as expected in a quasi-neutral plasma.

        Args:
            na: Species densities (batch, nx, ny, n_species)

        Returns:
            Charge-related momentum effect penalty (scalar)
        """
        # Charge tensor: (1, 1, 1, n_species)
        charges = torch.tensor(self.charges, dtype=torch.float32, device=na.device)
        charges = charges.view(1, 1, 1, -1)
        
        # Net charge density per cell: sum_species(Z_a * n_a) → (batch, nx, ny)
        # In a quasi-neutral plasma this should be spatially smooth.
        # Use relative variance (squared coefficient of variation) so the loss is
        # dimensionless and independent of the absolute density scale (~1e21 m^-3).
        charge_density = (charges * na).sum(dim=-1)
        
        charge_scale = charge_density.detach().abs().mean().clamp(min=1e-30)
        loss = torch.var(charge_density / charge_scale, dim=(-1, -2)).mean()
        
        return loss
    
    def compute_momentum_flux(self,
                             na: torch.Tensor,
                             ua: torch.Tensor) -> torch.Tensor:
        """
        Compute total outgoing momentum flux across all domain boundaries.

        Phi = sum_boundaries sum_alpha m_alpha * n_alpha * u_alpha^2 * delta_S

        Three outflow surfaces are considered:
          - Outer radial wall  (i = nx-1)       weighted by face_len_outer    (ny,)
          - Lower poloidal target (j = 0)       weighted by face_len_pol_low  (nx,)
          - Upper poloidal target (j = ny-1)    weighted by face_len_pol_high (nx,)

        u_a^2 is used (always positive) so the penalty is insensitive to the
        sign convention of the parallel velocity.

        Args:
            na: Species densities (batch, nx, ny, n_species)
            ua: Species velocities (batch, nx, ny, n_species)

        Returns:
            Total boundary momentum flux per sample (batch,)
        """
        masses = self.masses_kg.to(ua.device)
        if masses.dim() == 1:
            masses = masses.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Momentum flux density: m_a * n_a * u_a^2  (batch, nx, ny, n_species)
        flux_density = masses * na * (ua ** 2)

        # --- Outer radial boundary (i = nx-1) ---
        # face_len_outer: (ny,) → (1, ny, 1) for broadcasting over (batch, ny, n_species)
        face_outer = self.face_len_outer.to(ua.device).view(1, -1, 1)
        flux_outer = (flux_density[:, -1, :, :] * face_outer).sum(dim=(1, 2))  # (batch,)

        # --- Lower poloidal target (j = 0) ---
        # face_len_pol_low: (nx,) → (1, nx, 1) for broadcasting over (batch, nx, n_species)
        face_pol_low = self.face_len_pol_low.to(ua.device).view(1, -1, 1)
        flux_pol_low = (flux_density[:, :, 0, :] * face_pol_low).sum(dim=(1, 2))  # (batch,)

        # --- Upper poloidal target (j = ny-1) ---
        face_pol_high = self.face_len_pol_high.to(ua.device).view(1, -1, 1)
        flux_pol_high = (flux_density[:, :, -1, :] * face_pol_high).sum(dim=(1, 2))  # (batch,)

        return flux_outer + flux_pol_low + flux_pol_high
    
    def check_poloidal_periodicity(self,
                                  field: torch.Tensor,
                                  method: str = 'mse') -> torch.Tensor:
        """
        Check periodicity at poloidal boundaries (y-direction).
        Enforces continuity between last and first poloidal cells.
        
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
                na_pred: torch.Tensor,
                ua_pred: torch.Tensor,
                te_pred: torch.Tensor,
                na_true: Optional[torch.Tensor] = None,
                ua_true: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute momentum conservation loss.
        
        Args:
            na_pred: Predicted species densities (batch, nx, ny, n_species)
            ua_pred: Predicted species velocities (batch, nx, ny, n_species)
            te_pred: Predicted electron temperature (batch, nx, ny)
            na_true: Ground truth densities (optional)
            ua_true: Ground truth velocities (optional)
        
        Returns:
            Dictionary with loss components:
            - 'loss_total': Total momentum conservation loss
            - 'loss_momentum': Momentum balance term
            - 'loss_momentum_flux': Momentum flux conservation
            - 'loss_poloidal_periodicity': Poloidal circularity constraint
            - 'loss_charge_effects': Charge-dependent electromagnetic effects
        """
        # Compute total momentum
        momentum_pred = self.compute_total_momentum(na_pred, ua_pred)
        
        losses = {}
        
        if na_true is not None and ua_true is not None:
            # Compare with ground truth
            momentum_true = self.compute_total_momentum(na_true, ua_true)

            # Relative MSE: avoid absolute-SI-unit loss (~1e20) which creates gradient
            # magnitudes large enough to saturate clip_grad_norm_ and zero all updates.
            momentum_scale = momentum_true.abs().mean().clamp(min=1.0).detach()
            loss_momentum = F.mse_loss(momentum_pred / momentum_scale, momentum_true / momentum_scale)
            
            losses['loss_momentum'] = loss_momentum
        else:
            # Enforce momentum stability (minimize variance across batch)
            # This helps avoid pathological solutions with extreme velocities
            loss_momentum = torch.var(momentum_pred) * 0.1 + torch.mean(torch.abs(momentum_pred)) * 0.05
            
            losses['loss_momentum'] = loss_momentum
        
        # ============== MOMENTUM FLUX CONSERVATION ==============
        # Penalize excessive momentum leaving domain
        momentum_flux = self.compute_momentum_flux(na_pred, ua_pred)
        
        # Flux should be reasonable (not too large compared to total momentum)
        mean_momentum = torch.mean(torch.abs(momentum_pred))
        mean_momentum = torch.clamp(mean_momentum, min=1e-10)  # Avoid division by zero
        
        normalized_flux = momentum_flux / mean_momentum
        loss_momentum_flux = F.relu(torch.abs(normalized_flux) - 0.2).mean()
        
        losses['loss_momentum_flux'] = loss_momentum_flux
        
        # ============== POLOIDAL CIRCULARITY CONSTRAINTS ==============
        # Enforce periodicity for velocity and density
        loss_poloidal_na = self.check_poloidal_periodicity(na_pred, method='mse')
        loss_poloidal_ua = self.check_poloidal_periodicity(ua_pred, method='mse')
        
        loss_poloidal_periodicity = (loss_poloidal_na + loss_poloidal_ua) / 2.0
        
        losses['loss_poloidal_periodicity'] = loss_poloidal_periodicity
        
        # ============== CHARGE-DEPENDENT EFFECTS ==============
        # Account for electromagnetic forces on ions vs neutrals
        loss_charge_effects = self.compute_charge_effects(na_pred)
        
        losses['loss_charge_effects'] = loss_charge_effects
        
        # ============== TOTAL LOSS ==============
        loss_total = (
            self.weight_momentum * losses['loss_momentum'] +
            self.weight_momentum_flux * loss_momentum_flux +
            self.weight_poloidal_periodicity * loss_poloidal_periodicity +
            self.weight_charge_effects * loss_charge_effects
        )
        
        losses['loss_total'] = loss_total
        
        return losses


def momentum_conservation_loss(predictions: Dict[str, torch.Tensor],
                               targets: Optional[Dict[str, torch.Tensor]] = None,
                               data_handler = None,
                               **kwargs) -> torch.Tensor:
    """
    Convenience function for momentum conservation loss calculation.
    
    Args:
        predictions: Dictionary with predicted fields
                    Keys: 'na', 'ua', 'te' (all torch.Tensor)
        targets: Optional dictionary with ground truth fields
        data_handler: PlasmaDataHandler instance
        **kwargs: Additional arguments passed to MomentumConservationLoss
    
    Returns:
        Total momentum conservation loss
    """
    if data_handler is None:
        raise ValueError("data_handler must be provided")
    
    loss_fn = MomentumConservationLoss(data_handler, **kwargs)
    
    target_dict = targets or {}
    
    loss_dict = loss_fn.forward(
        na_pred=predictions['na'],
        ua_pred=predictions['ua'],
        te_pred=predictions['te'],
        na_true=target_dict.get('na'),
        ua_true=target_dict.get('ua'),
    )
    
    return loss_dict['loss_total']

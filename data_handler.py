"""
Data handler for plasma physics simulation data.
Manages data loading, storage, geometry, and normalization/denormalization.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


class PlasmaDataHandler:
    """
    Handler for plasma simulation data with normalization and geometry management.
    
    Attributes:
        - Stores simulation data (X, te, ti, na, ua, fnixap)
        - Stores geometry coordinates (crx, cry)
        - Stores normalization statistics (min/max values)
        - Provides methods for log-based normalization/denormalization
    """
    
    # Species names for particle densities and velocities
    SPECIES = ['D0', 'D1', 'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
    
    # Input parameter names
    INPUT_PARAMS = ['R', 'B', 'P', 'D_puff', 'N_puff', 'D_core', 'D_perp', 'chi_perp']
    
    # Which X parameters use log scale
    X_LOG_INDICES = [3, 4, 5]  # D_puff, N_puff, D_core indices (0-indexed)
    
    # Grid dimensions
    GRID_DIM = (104, 50)  # (nx, ny)
    N_SPECIES = 10
    
    def __init__(self, data_root: Union[str, Path] = 'a_dataset'):
        """
        Initialize the data handler.
        
        Args:
            data_root: Root directory containing the dataset
        """
        self.data_root = Path(data_root)
        
        # Data storage
        self.X: Optional[np.ndarray] = None          # (N_sim, 8) input parameters
        self.te: Optional[np.ndarray] = None         # (N_sim, 104, 50) electron temperature
        self.ti: Optional[np.ndarray] = None         # (N_sim, 104, 50) ion temperature
        self.na: Optional[np.ndarray] = None         # (N_sim, 104, 50, 10) species densities
        self.ua: Optional[np.ndarray] = None         # (N_sim, 104, 50, 10) species velocities
        self.fnixap: Optional[np.ndarray] = None     # (N_sim,) deuterium flux
        self.psol: Optional[np.ndarray] = None       # (N_sim,) power to SOL
        self.pwmxap: Optional[np.ndarray] = None     # (N_sim,) peak heatflux
        
        # Geometry
        self.crx: Optional[np.ndarray] = None        # (104, 50, 4) x-coordinates of grid corners
        self.cry: Optional[np.ndarray] = None        # (104, 50, 4) y-coordinates of grid corners
        
        # Normalization statistics
        self.norm_stats: Dict[str, np.ndarray] = {}
        
        # Current split (train/test)
        self.current_split = None
        self.split_size = 0
        
    def load_geometry(self) -> None:
        """Load geometry coordinates from files."""
        geom_path = self.data_root / 'geometry'
        self.crx = np.load(geom_path / 'crx.npy')
        self.cry = np.load(geom_path / 'cry.npy')
        logger.info(f"Loaded geometry: crx {self.crx.shape}, cry {self.cry.shape}")
    
    def load_normalization_stats(self) -> None:
        """Load normalization min/max statistics."""
        norm_path = self.data_root / 'norm_stats_minmax.npz'
        norm_data = np.load(norm_path)
        
        for key in norm_data.keys():
            self.norm_stats[key] = norm_data[key]
        
        logger.info(f"Loaded {len(self.norm_stats)} normalization parameters")
    
    def load_split(self, split: str = 'train') -> None:
        """
        Load data from a specific split (train or test).
        
        Args:
            split: 'train' or 'test'
        """
        if split not in ['train', 'test']:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        
        split_path = self.data_root / split
        
        # Load main fields
        self.X = np.load(split_path / 'X_tmp.npy')
        self.te = np.load(split_path / 'te_tmp.npy')
        self.na = np.load(split_path / 'na_tmp.npy')
        self.ua = np.load(split_path / 'ua_tmp.npy')
        self.fnixap = np.load(split_path / 'fnixap_tmp.npy')
        
        # Try to load optional fields
        ti_path = split_path / 'ti_tmp.npy'
        if ti_path.exists():
            self.ti = np.load(ti_path)
        
        psol_path = split_path / 'psol_tmp.npy'
        if psol_path.exists():
            self.psol = np.load(psol_path)
        
        pwmxap_path = split_path / 'pwmxap_tmp.npy'
        if pwmxap_path.exists():
            self.pwmxap = np.load(pwmxap_path)
        
        self.current_split = split
        self.split_size = len(self.X)
        
        logger.info(f"Loaded {split} split: {self.split_size} samples")
    
    def normalize(self, 
                  field: str, 
                  data: np.ndarray,
                  use_log: bool = True) -> np.ndarray:
        """
        Normalize data using min-max scaling (optionally log-based).
        
        For log-based fields (te, ti, na with log stats), applies log10 transformation first.
        For X field: X_min/max already contain log10 values for indices [3,4,5],
        so no extra transformation needed.
        
        Args:
            field: Field name ('X', 'te', 'ti', 'na', 'ua')
            data: Data to normalize
            use_log: If True, use log min/max for log-based normalization (if available)
        
        Returns:
            Normalized data in range [0, 1]
        """
        # Check if field exists in normalization stats
        if f"{field}_min" not in self.norm_stats:
            raise ValueError(f"Unknown field: {field}")
        
        data = data.astype(np.float64)
        vmin = self.norm_stats[f"{field}_min"].astype(np.float64)
        vmax = self.norm_stats[f"{field}_max"].astype(np.float64)
        
        # Special handling for X field with mixed log/linear scales
        if field == 'X':
            # X_min/max for indices [3,4,5] already store log10 values
            # So apply log10 to raw data at those indices to match stats scale
            data_to_norm = data.copy()
            for idx in self.X_LOG_INDICES:
                data_to_norm[..., idx] = np.log10(data[..., idx])
        elif field in ('te', 'ti', 'na') and use_log:
            # te/ti/na: NEW stats now store LN values directly in min/max
            # Apply natural log transformation to raw data
            data_to_norm = np.log(np.maximum(data, np.finfo(np.float64).tiny))
        elif field == 'ua':
            # ua: arcsinh transform (handles bipolar data better than linear)
            # vmin/vmax are arrays shape (10,) for each species
            # scale per-species: take max absolute value
            scale = np.maximum(np.abs(vmin), np.abs(vmax))  # shape (10,)
            scale = np.maximum(scale, 1.0)  # avoid division by zero
            data_to_norm = np.arcsinh(data / scale)
        else:
            # Other fields: linear scale
            data_to_norm = data
        
        # Normalize to [0, 1]
        if field == 'ua':
            # For ua: arcsinh is already applied, now scale from [-arcsinh(1), arcsinh(1)] to [0, 1]
            scale = np.maximum(np.abs(vmin), np.abs(vmax))  # shape (10,)
            scale = np.maximum(scale, 1.0)
            asinh_max = np.arcsinh(1.0)  # scalar ~1.8184
            normalized = (data_to_norm + asinh_max) / (2.0 * asinh_max)
        else:
            normalized = (data_to_norm - vmin) / (vmax - vmin)
        
        # Clamp to [0, 1] to handle data outside stats range
        # (can happen if data distribution changed or stats are outdated)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def denormalize(self,
                    field: str,
                    data: np.ndarray,
                    use_log: bool = True) -> np.ndarray:
        """
        Denormalize data from [0, 1] range back to original scale.
        
        For log-based fields, reverses log10 transformation.
        For X field, reverses selective log-scaling for specific parameters.
        
        Args:
            field: Field name ('X', 'te', 'ti', 'na', 'ua')
            data: Normalized data in range [0, 1]
            use_log: If True, use log min/max for log-based denormalization (if available)
        
        Returns:
            Denormalized data in original scale
        """
        # Check if field exists in normalization stats
        if f"{field}_min" not in self.norm_stats:
            raise ValueError(f"Unknown field: {field}")
        
        data = data.astype(np.float64)
        vmin = self.norm_stats[f"{field}_min"].astype(np.float64)
        vmax = self.norm_stats[f"{field}_max"].astype(np.float64)
        
        # Special handling for X field with mixed log/linear scales
        if field == 'X':
            # Denormalize from [0, 1]
            data_denorm = data * (vmax - vmin) + vmin
            
            # Convert log parameters back
            denormalized = data_denorm.copy()
            for idx in self.X_LOG_INDICES:
                denormalized[..., idx] = 10 ** data_denorm[..., idx]
            
            return denormalized
        elif field == 'ua':
            # ua: reverse arcsinh transform (vmin/vmax are arrays shape (10,))
            scale = np.maximum(np.abs(vmin), np.abs(vmax))
            scale = np.maximum(scale, 1.0)
            asinh_max = np.arcsinh(1.0)
            # Reverse: [0,1] → [-asinh_max, asinh_max] → sinh → original scale
            data_arcsinh = data * (2.0 * asinh_max) - asinh_max
            denormalized = scale * np.sinh(data_arcsinh)
            return denormalized
        elif field in ('te', 'ti', 'na') and use_log:
            # te/ti/na: NEW stats store LN values directly in min/max
            # Denormalize from [0, 1] to LN space
            data_denorm_log = data * (vmax - vmin) + vmin
            # Convert from LN back to linear scale
            denormalized = np.exp(data_denorm_log)
            return denormalized
        else:
            # Other fields: linear scale
            denormalized = data * (vmax - vmin) + vmin
            return denormalized
    
    def get_data_batch(self, indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Get a batch of data (optionally selected by indices).
        
        Args:
            indices: Indices to select from data. If None, return all data.
        
        Returns:
            Dictionary with keys: 'X', 'te', 'na', 'ua', 'fnixap', and optionally 'ti', 'psol', 'pwmxap'
        """
        batch = {}
        
        if indices is None:
            indices = slice(None)
        
        batch['X'] = self.X[indices]
        batch['te'] = self.te[indices]
        batch['na'] = self.na[indices]
        batch['ua'] = self.ua[indices]
        batch['fnixap'] = self.fnixap[indices]
        
        if self.ti is not None:
            batch['ti'] = self.ti[indices]
        if self.psol is not None:
            batch['psol'] = self.psol[indices]
        if self.pwmxap is not None:
            batch['pwmxap'] = self.pwmxap[indices]
        
        return batch
    
    def normalize_batch(self, 
                       batch: Dict[str, np.ndarray],
                       use_log: bool = True) -> Dict[str, np.ndarray]:
        """
        Normalize all fields in a batch.
        
        Args:
            batch: Dictionary with data to normalize
            use_log: If True, use log min/max normalization
        
        Returns:
            Dictionary with normalized data
        """
        normalized_batch = {}
        
        for field in batch.keys():
            try:
                normalized_batch[field] = self.normalize(field, batch[field], use_log=use_log)
            except ValueError:
                # Field not in normalization stats, keep as is
                normalized_batch[field] = batch[field]
        
        return normalized_batch
    
    def denormalize_batch(self,
                         batch: Dict[str, np.ndarray],
                         use_log: bool = True) -> Dict[str, np.ndarray]:
        """
        Denormalize all fields in a batch.
        
        Args:
            batch: Dictionary with normalized data
            use_log: If True, use log min/max denormalization
        
        Returns:
            Dictionary with denormalized data in original scale
        """
        denormalized_batch = {}
        
        for field in batch.keys():
            try:
                denormalized_batch[field] = self.denormalize(field, batch[field], use_log=use_log)
            except ValueError:
                # Field not in normalization stats, keep as is
                denormalized_batch[field] = batch[field]
        
        return denormalized_batch
    
    def get_species_density(self, sample_idx: int, species_idx: int) -> np.ndarray:
        """
        Get density field for a specific species and sample.
        
        Args:
            sample_idx: Sample index
            species_idx: Species index (0-9)
        
        Returns:
            Density field (104, 50)
        """
        if species_idx < 0 or species_idx >= self.N_SPECIES:
            raise ValueError(f"species_idx must be in range [0, {self.N_SPECIES-1}]")
        
        return self.na[sample_idx, :, :, species_idx]
    
    def get_species_velocity(self, sample_idx: int, species_idx: int) -> np.ndarray:
        """
        Get velocity field for a specific species and sample.
        
        Args:
            sample_idx: Sample index
            species_idx: Species index (0-9)
        
        Returns:
            Velocity field (104, 50)
        """
        if species_idx < 0 or species_idx >= self.N_SPECIES:
            raise ValueError(f"species_idx must be in range [0, {self.N_SPECIES-1}]")
        
        return self.ua[sample_idx, :, :, species_idx]
    
    def summary(self) -> str:
        """Get a summary of loaded data."""
        summary_lines = [
            "PlasmaDataHandler Summary:",
            "=" * 50,
        ]
        
        if self.current_split:
            summary_lines.append(f"Current split: {self.current_split}")
            summary_lines.append(f"Number of samples: {self.split_size}")
            summary_lines.append("")
            
            if self.X is not None:
                summary_lines.append(f"X shape: {self.X.shape}")
            if self.te is not None:
                summary_lines.append(f"te (electron temp): {self.te.shape}")
            if self.ti is not None:
                summary_lines.append(f"ti (ion temp): {self.ti.shape}")
            if self.na is not None:
                summary_lines.append(f"na (species densities): {self.na.shape}")
            if self.ua is not None:
                summary_lines.append(f"ua (species velocities): {self.ua.shape}")
            if self.fnixap is not None:
                summary_lines.append(f"fnixap (deuterium flux): {self.fnixap.shape}")
        
        if self.crx is not None:
            summary_lines.append(f"\nGeometry crx: {self.crx.shape}")
            summary_lines.append(f"Geometry cry: {self.cry.shape}")
        
        if self.norm_stats:
            summary_lines.append(f"\nNormalization stats: {len(self.norm_stats)} parameters")
        
        summary_lines.append(f"\nSpecies: {', '.join(self.SPECIES)}")
        summary_lines.append(f"Input parameters: {', '.join(self.INPUT_PARAMS)}")
        
        return "\n".join(summary_lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create handler and load data
    handler = PlasmaDataHandler(data_root='a_dataset')
    handler.load_geometry()
    handler.load_normalization_stats()
    handler.load_split(split='train')
    
    print(handler.summary())
    print()
    
    # Example: Get and normalize a batch
    batch = handler.get_data_batch(indices=np.arange(10))
    print(f"Original batch X range: [{batch['X'].min():.3e}, {batch['X'].max():.3e}]")
    
    normalized_batch = handler.normalize_batch(batch)
    print(f"Normalized batch X range: [{normalized_batch['X'].min():.3f}, {normalized_batch['X'].max():.3f}]")
    
    denormalized_batch = handler.denormalize_batch(normalized_batch)
    print(f"Denormalized batch X range: [{denormalized_batch['X'].min():.3e}, {denormalized_batch['X'].max():.3e}]")
    print(f"Reconstruction error: {np.max(np.abs(batch['X'] - denormalized_batch['X'])):.3e}")

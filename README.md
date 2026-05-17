# Physics-Informed Parameter-Conditional VAE for Tokamak Edge Plasma

This repository contains a CNN-based Parameter-Conditional VAE (PCVAE) for modeling and generating tokamak edge plasma states with geometry-aware physics losses.

## What it does

The model learns the distribution of plasma fields conditioned on tokamak operating parameters. Physical parameters are concatenated with the latent code in the bottleneck and decoder, so the encoder stays condition-free while the generative path is parameter-aware.

The model handles:

- Spatial fields: electron and ion temperature, species densities, parallel velocities, and integrated ion flux
- Conditioning: 8 physical input parameters
- Geometry: curvilinear tokamak mesh with cell-area weighting and metric-aware derivatives
- Physics losses: energy, continuity, momentum, and global balance constraints

## Main Features

- Geometry-aware losses computed on the actual tokamak grid
- 2D continuity and momentum residuals in both coordinate directions
- Weak conditioning in the bottleneck and decoder only
- Native support for the non-uniform curvilinear mesh
- Visualization scripts for per-field and per-species inspection

## Installation

Requirements:

- Python 3.8+
- PyTorch 1.9+ (CUDA optional)
- NumPy, Matplotlib, tqdm

Setup:

```bash
git clone <repo_url>
cd plasma_refactor

python -m venv venv
venv\Scripts\activate

pip install torch numpy matplotlib tqdm
```

## Dataset

Place the dataset in `a_dataset/` with the following structure:

```text
a_dataset/
├── norm_stats_minmax.npz
├── geometry/
│   ├── crx.npy
│   └── cry.npy
├── train/
│   ├── X_tmp.npy
│   ├── te_tmp.npy, ti_tmp.npy
│   ├── na_tmp.npy, ua_tmp.npy
│   └── fnixap_tmp.npy
└── test/
  └── same files as train/
```

Data dimensions:

- Grid: 104 x 50
- Species: 10
- Input parameters: 8

## Training

```bash
python train_pcvae.py --epochs 150 --results_dir train_PCVAE_results
```

Key defaults in `train_pcvae.py`:

| Parameter | Default | Description |
|---|---:|---|
| `BATCH_SIZE` | 64 | Batch size |
| `LR` | 1e-3 | Adam learning rate |
| `LATENT_SIZE` | 128 | Latent dimension |
| `BETA_KLD` | 0.01 | KL weight |
| `FREE_BITS` | 0.5 | Free bits per latent dimension |
| `LAMBDA_ENERGY` | 1e-3 | Energy loss weight |
| `LAMBDA_CONTINUITY` | 1e-3 | Continuity loss weight |
| `LAMBDA_NAVIER_STOKES` | 1e-3 | Momentum loss weight |
| `LAMBDA_GEOMETRY_BALANCE` | 5e-4 | Global balance weight |

Training logs report:

- `loss`: total loss
- `mse`: reconstruction error
- `kld`: raw KL divergence
- `E`, `cont`, `ns`: physics residuals
- `gb`: global geometry-balance residual
- `val`: validation MSE on the test set

The default checkpoint is saved as `train_PCVAE_results/best_PCVAE.pt`.

## Visualization

Use the visualization script to compare reconstructions with ground truth:

```bash
# Random samples
python visualize_pcvae.py --checkpoint train_PCVAE_results/best_PCVAE.pt --n 4

# Specific samples
python visualize_pcvae.py --checkpoint train_PCVAE_results/best_PCVAE.pt --sample 42 130 642

# More Monte Carlo averages per cell
python visualize_pcvae.py --checkpoint train_PCVAE_results/best_PCVAE.pt --sample 0 --k 20 --out_dir figs_PCVAE_detailed
```

Typical outputs:

- `fig1_Te_Ti.png` - Electron and ion temperature comparison
- `fig2_n_tot_ne.png` - Total density and electron density
- `fig3_density_s*.png` - Per-species densities
- `fig4_velocity_s*.png` - Parallel velocities
- `fig5_ns_div_s*.png` - Momentum-flux divergence
- `fig6_fnixap.png` - Integrated ion flux correlation

## Model Architecture

This is a weakly conditional VAE: the encoder processes only plasma fields, while the bottleneck and decoder receive the normalized conditioning parameters.

Encoder:

1. Initial convolution: 23 -> 32 channels + residual block
2. Three downsampling blocks with residual connections
3. Flattened feature map: 256 x 13 x 6
4. Outputs mean and log-variance for a 128-dimensional latent space

Bottleneck and decoder:

1. Concatenate latent vector and normalized parameters
2. Linear projection to 512, then to 256 x 13 x 6
3. Three upsampling blocks with skip connections
4. Final convolution back to 23 channels with sigmoid output

Input/output tensor shape:

- `(B, 23, 104, 50)`
- 23 channels = 1 Te + 1 Ti + 10 densities + 10 velocities + 1 integrated ion flux
- 8 conditioning parameters: `R`, `B`, `P`, `D_puff`, `N_puff`, `D_core`, `D_perp`, `chi_perp`

## Physics Losses

All spatial derivatives are computed on the physical tokamak grid using center-to-center distances and cell-area weights.

### Energy loss

Compares gradients of energy density between reconstruction and target:

$$
L_E = \mathrm{mean}\left(|\nabla E_{recon} - \nabla E_{target}| \cdot area\right)
$$

### Continuity loss

Penalizes differences in species flux divergence:

$$
L_C = \mathrm{mean}\left(|\nabla \cdot (n u)_{recon} - \nabla \cdot (n u)_{target}| \cdot area\right)
$$

### Navier-Stokes loss

Penalizes differences in momentum-flux divergence:

$$
L_{NS} = \mathrm{mean}\left(|\nabla \cdot (n u^2 + nT)_{recon} - \nabla \cdot (n u^2 + nT)_{target}| \cdot area\right)
$$

### Geometry balance loss

Encourages global agreement after integration over the full mesh:

$$
L_{gb} = \frac{1}{3}\left(\frac{|E_{int,r}|}{|E_{int,t}|} + \frac{|j_{int,r}|}{|j_{int,t}|} + \frac{|m_{int,r}|}{|m_{int,t}|}\right)
$$

## Project Structure

```text
.
├── train_pcvae.py
├── visualize_pcvae.py
├── README.md
├── a_dataset/
├── train_PCVAE_results/
│   └── best_PCVAE.pt
└── figs_PCVAE/
```

## Notes

- The current implementation uses conditioning only in the bottleneck and decoder.
- The normalization file `a_dataset/norm_stats_minmax.npz` must match the dataset version used for training.
- The scripts expect the geometry files `crx.npy` and `cry.npy` to be available under `a_dataset/geometry/`.

## Citation

//

## License

//

## Status

Active development

# Physics-Informed Parameter-Conditional VAE for Tokamak Edge Plasma

A deep learning framework for probabilistic modeling and generation of edge plasma states in tokamak divertors using a **Parameter-Conditional Variational Autoencoder** with geometry-aware physics losses.

## Overview

This project implements a CNN-based Parameter-Conditional VAE to learn the distribution of plasma fields conditioned on tokamak operating parameters. The model concatenates physical parameters with the latent code in the bottleneck and decoder, enabling flexible generation of diverse plasma states under different operating conditions. The model incorporates:

- **Spatial fields**: Electron/ion temperature (Te, Ti), density per species (n_s), parallel velocity (u_s), integrated ion flux (Γ_ix)
- **Conditioning**: 8 physical parameters (R, B, P, gas puff rates, transport coefficients)
- **Geometry**: Full curvilinear tokamak mesh with cell areas and metric-aware discretization
- **Physics losses**: Energy conservation, particle continuity, momentum balance with geometry weighting
- **Regularization**: Free-bits KL regularization and global cell-integrated balance constraints

## Key Features

✅ **Geometry-aware physics losses**: Spatial derivatives computed with real center-to-center distances and cell area weighting  
✅ **Full 2D residuals**: Continuity and Navier-Stokes equations matched in both coordinate directions  
✅ **Parameter conditioning**: Physical parameters (R, B, P, gas puff, transport) influence latent space and reconstruction  
✅ **Curvilinear mesh support**: Native handling of non-uniform tokamak grids  
✅ **Visualization pipeline**: Interactive plots with geometric overlay and per-species metrics  

## Installation

### Requirements
- Python 3.8+
- PyTorch >= 1.9 (CUDA optional but recommended)
- NumPy, Matplotlib, tqdm

### Setup

```bash
# Clone repository
git clone <repo_url>
cd refactor_2

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib tqdm
```

## Dataset

Place the dataset in `a_dataset/`:

```
a_dataset/
├── norm_stats_minmax.npz          # Normalization statistics
├── geometry/
│   ├── crx.npy                    # X coordinates of cell corners (NX, NY, 4)
│   └── cry.npy                    # Y coordinates of cell corners (NX, NY, 4)
├── train/
│   ├── X_tmp.npy                  # Operating parameters (N_train, 8)
│   ├── te_tmp.npy, ti_tmp.npy     # Temperature fields (N_train, NX, NY)
│   ├── na_tmp.npy, ua_tmp.npy     # Density and velocity per species
│   └── fnixap_tmp.npy             # Integrated ion flux
└── test/
    └── [same structure as train]
```

**Data format:**
- NX=104, NY=50, NS=10 (species)
- Temperature: Joules → eV on denormalization
- Density: m⁻³, velocity: m/s

## Usage

### Training

```bash
python train_cvae.py --epochs 150 --results_dir train_cvae_results
```

**Key hyperparameters** (in `train_cvae.py`):
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `BATCH_SIZE` | 64 | Training batch size |
| `LR` | 1e-3 | Adam learning rate |
| `LATENT_SIZE` | 128 | Latent space dimension |
| `BETA_KLD` | 0.01 | KL divergence coefficient |
| `FREE_BITS` | 0.5 | Free bits per dimension (nats) |
| `LAMBDA_ENERGY` | 1e-3 | Energy loss weight |
| `LAMBDA_CONTINUITY` | 1e-3 | Continuity loss weight |
| `LAMBDA_NAVIER_STOKES` | 1e-3 | Momentum loss weight |
| `LAMBDA_GEOMETRY_BALANCE` | 5e-4 | Global balance loss weight |

**Training loop metrics:**
- `loss`: Total loss per sample
- `mse`: Reconstruction MSE
- `kld`: Raw KL divergence (unpenalized)
- `E`, `cont`, `ns`: Per-species physics residuals
- `gb`: Global geometry balance residual
- `val`: Validation MSE on test set

### Visualization

Generate plots from a trained checkpoint:

```bash
# Random 4 samples
python visualize_cvae.py --checkpoint train_cvae_results/best_cvae.pt --n 4

# Specific samples
python visualize_cvae.py --checkpoint train_cvae_results/best_cvae.pt --sample 42 130 642

# Average over K=20 prior samples instead of K=10
python visualize_cvae.py --checkpoint train_cvae_results/best_cvae.pt --sample 0 --k 20 --out_dir figs_cvae_detailed
```

**Output files:**
- `fig1_Te_Ti.png` — Electron/ion temperature GT vs reconstruction
- `fig2_n_tot_ne.png` — Total density and electron density
- `fig3_density_s*.png` — Per-species densities (all 10 species)
- `fig4_velocity_s*.png` — D⁺ and N⁺ parallel velocities
- `fig5_ns_div_s*.png` — Momentum flux divergence in normalized space
- `fig6_fnixap.png` — Integrated ion flux correlation (full test set)

All plots use curvilinear tokamak cell geometry with physical colorscales.

## Model Architecture

This is a **weakly-conditional VAE** where physical parameters are concatenated **only in the bottleneck and decoder**, not in the encoder. This design allows the model to learn a shared representation across all conditions, with conditioning applied only to the generative process.

### Encoder
1. Initial Conv (23 → 32 channels) + ResBlock
2. 3× DownsampleBlock (stride-2 with residuals) — operates on spatial data only
3. Flatten → Dense representation (256×13×6)
4. Output: μ(z), log σ(z) ∈ ℝ¹²⁸ (condition-free)

### Bottleneck & Decoder  
1. Concatenate latent z and normalized parameters c: [z || c]
2. Linear: (z || c) → 512 → 256×13×6
3. 3× UpsampleBlock (stride-2 with zero-filled skip connections)
4. Final Conv (64 → 23 channels, Sigmoid output)

**Input/Output:** (B, 23, 104, 50) where 23 = 1(Te) + 1(Ti) + 10(n_s) + 10(u_s) + 1(Γ_ix)  
**Conditioning:** 8 physical parameters (R, B, P, D_puff, N_puff, D_core, D_perp, χ_perp) min-max normalized and concatenated in decoder


## Physics Losses (Kary fizyczne)

Model wykorzystuje cztery główne kary fizyczne, które wymuszają zgodność rekonstrukcji z prawami fizyki plazmy. Każda z nich jest liczona na siatce geometrycznej tokamaka, z uwzględnieniem metryki i powierzchni komórek.

### 1. Energy Loss (Kara energetyczna)
Porównuje gęstość energii (termodynamicznej i kinetycznej) pomiędzy rekonstrukcją a danymi referencyjnymi. Różnica liczona jest pochodną przestrzenną (gradientem) i ważona powierzchnią komórek.

```
L_E = mean( |∇E_recon - ∇E_target| * area )
```
- E — gęstość energii (termodynamiczna + kinetyczna)
- ∇ — pochodna przestrzenna liczona na siatce zakrzywionej
- area — powierzchnia komórki (waga geometryczna)

**Znaczenie:** Model uczy się nie tylko wartości energii, ale także poprawnych przepływów i rozkładów przestrzennych.

### 2. Continuity Loss (Kara ciągłości)
Wymusza zachowanie ciągłości cząstek (liczby cząstek nie może "znikać" ani "powstawać" w siatce). Liczona jest dywergencja strumienia cząstek n_s u_s dla każdego gatunku.

```
L_C = mean( |div(nu)_recon - div(nu)_target| * area )
```
- n_s — gęstość cząstek dla gatunku s
- u_s — prędkość równoległa dla gatunku s
- div — dywergencja (pochodna przestrzenna)
- area — powierzchnia komórki

**Znaczenie:** Model uczy się zachowywać liczbę cząstek w każdym punkcie siatki.

### 3. Navier-Stokes Loss (Kara pędu)
Wymusza zgodność z równaniem pędu (Naviera-Stokesa) — czyli poprawny przepływ pędu, uwzględniając ciśnienie i ruch.

```
L_NS = mean( |div(nu^2 + nT)_recon - div(nu^2 + nT)_target| * area )
```
- n — gęstość
- u — prędkość
- T — temperatura jonowa
- div — dywergencja
- area — powierzchnia komórki

**Znaczenie:** Model uczy się poprawnych przepływów pędu i rozkładów ciśnienia.

### 4. Geometry Balance Loss (Kara globalnej równowagi geometrycznej)
Wymusza globalną zgodność bilansów (energia, cząstki, pęd) po całkowaniu po całej siatce, z wagą powierzchni.

```
L_gb = ( |E_int_r|/|E_int_t| + |j_int_r|/|j_int_t| + |m_int_r|/|m_int_t| ) / 3
```
- E_int_r — całkowita energia z rekonstrukcji
- E_int_t — całkowita energia z danych
- j_int_r — całkowity strumień cząstek z rekonstrukcji
- j_int_t — całkowity strumień cząstek z danych
- m_int_r — całkowity pęd z rekonstrukcji
- m_int_t — całkowity pęd z danych

**Znaczenie:** Model nie tylko lokalnie, ale i globalnie zachowuje prawa fizyki.

**Wszystkie pochodne przestrzenne** liczone są na fizycznej siatce tokamaka:
- dx, dy: odległości środek-środek z krzywoliniowych współrzędnych
- area_x, area_y: uśrednione powierzchnie komórek do ważenia strumieni

## File Structure

```
.
├── train_cvae.py              # Training script (main entry point)
├── visualize_cvae.py          # Visualization script
├── cvae_original.py           # Original MLP-based CVAE (deprecated)
├── a_dataset/                 # Data directory (not in repo)
│   ├── geometry/
│   ├── train/, test/
│   └── norm_stats_minmax.npz
├── train_cvae_results/        # Checkpoint output
│   └── best_cvae.pt
└── figs_cvae/                 # Visualization output
    ├── fig1_Te_Ti.png
    └── [...]
```

## Performance Metrics

The model is evaluated on the test set using:
1. **Reconstruction MSE**: normalized per field element
2. **Physics residuals**: Energy, continuity, momentum per species
3. **Global balance**: Area-integrated conservation checks
4. **Per-field statistics**: RMSE and mean relative error for Te, Ti, n_s, u_s, Γ_ix

Typical metrics after 100 epochs:
- Test MSE: ~0.03–0.05
- Energy residual: ~0.001–0.01
- Continuity residual: ~0.005–0.02

## Advanced Options

### Free Bits Regularization
Prevents posterior collapse by allowing KL ≤ FREE_BITS per dimension:
$$\mathcal{L}_{KL} = \beta_{KLD} \sum_d \max(0, KL_d - \text{FREE_BITS})$$

### Physics Warmup
Physics losses scale gradually from 0 to 1 over PHYS_WARMUP epochs to stabilize training.

### Cosine Annealing
Learning rate follows cosine decay from LR to 1e-6 over full training.

## Citation

If you use this code, please cite:

```bibtex
@software{pcvae_tokamak_2026,
  title={Physics-Informed Parameter-Conditional VAE for Tokamak Edge Plasma},
  author={[Your Name]},
  year={2026},
  url={https://github.com/<repo>}
}
```

## References

- Kingma & Welling (2013): *Auto-Encoding Variational Bayes* [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Sohn et al. (2015): *Learning Structured Output Representation using Deep Conditional Generative Models* [NeurIPS]
- Free bits regularization from Burda et al. (2015): *Importance weighted autoencoders* [ICLR]

## License

[Specify license, e.g., MIT, Apache 2.0]

## Contact & Support

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Last updated:** May 2026  
**Status:** Active development

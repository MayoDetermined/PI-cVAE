### Parameters
n_sim - number of simulations
nx - grid dimension along the torus
ny - grid dimension radial direction
ns - particle (D0,D1,N0,N1,N2,N3,...,N7)
 
Dimensions:
X - (N_sim,8) 
te - (N_sim,nx,ny)
na - (N_sim,nx,ny,ns)

The files crx.npy / cry.npy contain the x and y coordinates of the 4 corners of each grid cell in the computational grid.

### Model fields
The data contains  2D fields on the 104x50 computational grid:
- `item='te'` Electron temperature $\mathrm{J}$
- `item='ti'` Ion temperature $\mathrm{J}$

- `item='na',species='D0'` Deuterium neutral gas density $\mathrm{m}^{-3}$
- `item='na',species='D1'` Deuterium ion density $\mathrm{m}^{-3}$
- `item='na',species='N0'` Nitrogen neutral gas density $\mathrm{m}^{-3}$
- `item='na',species='N1'` Nitrogen $\mathrm{N}^{+1}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N2'` Nitrogen $\mathrm{N}^{+2}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N3'` Nitrogen $\mathrm{N}^{+3}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N4'` Nitrogen $\mathrm{N}^{+4}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N5'` Nitrogen $\mathrm{N}^{+5}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N6'` Nitrogen $\mathrm{N}^{+6}$ ion density $\mathrm{m}^{-3}$
- `item='na',species='N7'` Nitrogen $\mathrm{N}^{+7}$ ion density $\mathrm{m}^{-3}$

- `item='ua',species='D0'` Deuterium neutral gas parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='D1'` Deuterium ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N0'` Nitrogen neutral gas parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N1'` Nitrogen $\mathrm{N}^{+1}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N2'` Nitrogen $\mathrm{N}^{+2}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N3'` Nitrogen $\mathrm{N}^{+3}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N4'` Nitrogen $\mathrm{N}^{+4}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N5'` Nitrogen $\mathrm{N}^{+5}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N6'` Nitrogen $\mathrm{N}^{+6}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$
- `item='ua',species='N7'` Nitrogen $\mathrm{N}^{+7}$ ion parallel velocity $\mathrm{m}/\mathrm{s}$

Additionally the model can predict the following scalar quantities
- `item='pwmxap'` Peak heatflux at the outer (LFS) divertor target $\mathrm{W}/\mathrm{m}^2$
- `item='fnixap'` Integrated deuterium ion flux to the outer (LFS) divertor target  $\mathrm{atoms}/\mathrm{s}$
- `item='psol'` Power crossing the separtrix from core to SOL plasma $\mathrm{W}$

The eight input parameters:
- $R$ Tokamak major radius $\mathrm{m}$
- $B$ Toroidal magnetic field strength (on axis) $\mathrm{T}$
- $P$ Input power into simulation domain $\mathrm{W}$
- $D_\mathrm{puff}$ Deuterium gas puff rate $\mathrm{atoms}/\mathrm{s}$
- $N_\mathrm{puff}$ Nitrogen gas puff rate $\mathrm{atoms}/\mathrm{s}$
- $D_\mathrm{core}$ Deuterium core fueling rate $\mathrm{atoms}/\mathrm{s}$
- $D_\perp$ Cross-field particle transport coefficient $\mathrm{m}^2/\mathrm{s}$
- $\chi_\perp$ Cross-field heat transport coefficient  $\mathrm{m}^2/\mathrm{s}$

### Parameter range
Simulations span the following parameter ranges:

| Parameter         | Range                                      |
| ----------------- | ------------------------------------------ |
| $R$               | $1-10\mathrm{m}$                           |
| $B$               | $1-10\mathrm{T}$                           |
| $P$               | $10-200\mathrm{MW}$                        |
| $D_\mathrm{puff}$ | $10^{18}-10^{24}\mathrm{atoms}/\mathrm{s}$ |
| $N_\mathrm{puff}$ | $10^{18}-10^{23}\mathrm{atoms}/\mathrm{s}$ |
| $D_\mathrm{core}$ | $10^{19}-10^{24}\mathrm{atoms}/\mathrm{s}$ |
| $D_\perp$         | $0.1-2\mathrm{m}^2/\mathrm{s}$             |
| $\chi_\perp$      | $0.1-2\mathrm{m}^2/\mathrm{s}$             |

import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader_kwargs = {'num_workers': 0, 'pin_memory': torch.cuda.is_available()}

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
batch_size  = 64
latent_size = 128
epochs      = 150
lr          = 1e-4   # lower LR for large input
beta            = 1e-2   # final KLD weight; annealed from 0 during warmup
beta_warmup     = 30     # epochs over which beta ramps up linearly
lambda_spectral = 0.01   # weight for spectral (FFT) loss
lambda_physics  = 1e-3   # weight for thermal energy conservation loss

# Grid / feature dimensions
NX, NY, NS = 104, 50, 10
FIELD_SIZE = (2 + 2*NS) * NX * NY + 1  # te + ti + na(NS) + ua(NS) + fnixap(scalar)
COND_SIZE  = 8                           # 8 physical input parameters

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
# Load pre-computed stats (min/max in ln-space for fields, raw for X).
_stats = np.load(os.path.join('a_dataset', 'norm_stats_minmax.npz'))

# X: per-column min/max (some columns already stored as log10 values in data)
_X_min = torch.tensor(_stats['X_min'], dtype=torch.float32)   # (8,)
_X_max = torch.tensor(_stats['X_max'], dtype=torch.float32)   # (8,)

# Fields: min/max of ln(field)
_te_ln_min = float(_stats['te_min'])
_te_ln_max = float(_stats['te_max'])
_ti_ln_min = float(_stats['ti_min'])
_ti_ln_max = float(_stats['ti_max'])

# na: log-space min/max per species (shape (NS,))
_na_ln_min = torch.tensor(_stats['na_min'], dtype=torch.float32)  # (NS,)
_na_ln_max = torch.tensor(_stats['na_max'], dtype=torch.float32)  # (NS,)

# ua: linear min/max per species (shape (NS,))
_ua_min = torch.tensor(_stats['ua_min'], dtype=torch.float32)  # (NS,)
_ua_max = torch.tensor(_stats['ua_max'], dtype=torch.float32)  # (NS,)

# fnixap: log-space min/max computed from training data
_fnixap_train = np.load(os.path.join('a_dataset', 'train', 'fnixap_tmp.npy'))
_fnixap_ln_min = float(np.log(_fnixap_train.min()))
_fnixap_ln_max = float(np.log(_fnixap_train.max()))

# Species masses [kg]: D0, D1, N0, N1, ..., N7
_M_D  = 2.0  * 1.67262192e-27
_M_N  = 14.0 * 1.67262192e-27
_MASS = torch.tensor([_M_D, _M_D,
                       _M_N, _M_N, _M_N, _M_N, _M_N, _M_N, _M_N, _M_N],
                      dtype=torch.float32)  # (NS,)


def normalize_X(X: torch.Tensor) -> torch.Tensor:
    """Map each X column from its data range to [0, 1]."""
    xmin = _X_min.to(X.device)
    xmax = _X_max.to(X.device)
    return (X - xmin) / (xmax - xmin)


def _log_minmax(field: torch.Tensor, ln_min: float, ln_max: float) -> torch.Tensor:
    """ln → min-max → [0, 1]."""
    return (torch.log(field.clamp(min=1e-40)) - ln_min) / (ln_max - ln_min)


def normalize_fields(te: torch.Tensor, ti: torch.Tensor,
                     na: torch.Tensor, ua: torch.Tensor,
                     fnixap: torch.Tensor) -> torch.Tensor:
    """Normalize all fields to [0,1] and return as a flat (B, FIELD_SIZE) tensor.
    Layout: [te | ti | na_s0..s9 | ua_s0..s9 | fnixap], each 2D block is (NX*NY).
    """
    te_n = _log_minmax(te, _te_ln_min, _te_ln_max).view(-1, NX * NY)
    ti_n = _log_minmax(ti, _ti_ln_min, _ti_ln_max).view(-1, NX * NY)

    # na: (B, NX, NY, NS) — log-norm per species
    na_ln = torch.log(na.clamp(min=1e-40))                          # (B, NX, NY, NS)
    na_mn = _na_ln_min.to(na.device)                                # (NS,)
    na_mx = _na_ln_max.to(na.device)
    na_n  = (na_ln - na_mn) / (na_mx - na_mn)                       # (B, NX, NY, NS)
    na_n  = na_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)      # (B, NS*NX*NY)

    # ua: (B, NX, NY, NS) — linear min-max per species
    ua_mn = _ua_min.to(ua.device)
    ua_mx = _ua_max.to(ua.device)
    ua_n  = (ua - ua_mn) / (ua_mx - ua_mn)                          # (B, NX, NY, NS)
    ua_n  = ua_n.permute(0, 3, 1, 2).reshape(-1, NS * NX * NY)      # (B, NS*NX*NY)

    # fnixap: (B,) — log-space min-max
    fnixap_n = (_log_minmax(fnixap.view(-1, 1), _fnixap_ln_min, _fnixap_ln_max))  # (B, 1)

    return torch.cat([te_n, ti_n, na_n, ua_n, fnixap_n], dim=1)


def denormalize_fields(x_flat: torch.Tensor):
    """Invert normalize_fields; returns (te, ti, na, ua, fnixap) in physical units."""
    split = [NX*NY, NX*NY, NS*NX*NY, NS*NX*NY, 1]
    te_n, ti_n, na_n, ua_n, fnixap_n = torch.split(x_flat, split, dim=1)

    te = torch.exp(te_n * (_te_ln_max - _te_ln_min) + _te_ln_min).view(-1, NX, NY)
    ti = torch.exp(ti_n * (_ti_ln_max - _ti_ln_min) + _ti_ln_min).view(-1, NX, NY)

    na_mn = _na_ln_min.to(x_flat.device)
    na_mx = _na_ln_max.to(x_flat.device)
    na_n  = na_n.view(-1, NS, NX, NY).permute(0, 2, 3, 1)           # (B, NX, NY, NS)
    na    = torch.exp(na_n * (na_mx - na_mn) + na_mn)

    ua_mn = _ua_min.to(x_flat.device)
    ua_mx = _ua_max.to(x_flat.device)
    ua_n  = ua_n.view(-1, NS, NX, NY).permute(0, 2, 3, 1)           # (B, NX, NY, NS)
    ua    = ua_n * (ua_mx - ua_mn) + ua_mn

    fnixap = torch.exp(fnixap_n * (_fnixap_ln_max - _fnixap_ln_min) + _fnixap_ln_min).view(-1)

    return te, ti, na, ua, fnixap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SimDataset(torch.utils.data.Dataset):
    def __init__(self, split: str):
        base = os.path.join('a_dataset', split)
        self.X  = torch.tensor(np.load(os.path.join(base, 'X_tmp.npy')),  dtype=torch.float32)
        self.te = torch.tensor(np.load(os.path.join(base, 'te_tmp.npy')), dtype=torch.float32)
        self.ti = torch.tensor(np.load(os.path.join(base, 'ti_tmp.npy')), dtype=torch.float32)
        self.na     = torch.tensor(np.load(os.path.join(base, 'na_tmp.npy')),     dtype=torch.float32)
        self.ua     = torch.tensor(np.load(os.path.join(base, 'ua_tmp.npy')),     dtype=torch.float32)
        self.fnixap = torch.tensor(np.load(os.path.join(base, 'fnixap_tmp.npy')), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.te[idx], self.ti[idx], self.na[idx], self.ua[idx], self.fnixap[idx]


train_loader = torch.utils.data.DataLoader(
    SimDataset('train'), batch_size=batch_size, shuffle=True, **loader_kwargs)
test_loader = torch.utils.data.DataLoader(
    SimDataset('test'), batch_size=batch_size, shuffle=False, **loader_kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CVAE(nn.Module):
    def __init__(self, feature_size: int, latent_size: int, class_size: int):
        super().__init__()
        self.feature_size = feature_size
        self.class_size   = class_size

        # Encoder: Q(z | x, c)
        self.enc_fc1    = nn.Linear(feature_size + class_size, 2048)
        self.enc_fc2    = nn.Linear(2048, 1024)
        self.enc_mu     = nn.Linear(1024, latent_size)
        self.enc_logvar = nn.Linear(1024, latent_size)

        # Decoder: P(x | z, c)
        self.dec_fc1 = nn.Linear(latent_size + class_size, 1024)
        self.dec_fc2 = nn.Linear(1024, 2048)
        self.dec_out = nn.Linear(2048, feature_size)

        self.act = nn.ELU()

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.act(self.enc_fc1(torch.cat([x, c], dim=1)))
        h = self.act(self.enc_fc2(h))
        mu     = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(-10, 4)  # prevent KLD explosion
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.act(self.dec_fc1(torch.cat([z, c], dim=1)))
        h = self.act(self.dec_fc2(h))
        return torch.sigmoid(self.dec_out(h))   # output in [0, 1]

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


model     = CVAE(FIELD_SIZE, latent_size, COND_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def spectral_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between normalized 2D FFT magnitude spectra of te and ti.
    Amplitudes are divided by the mean target magnitude so the loss is O(1),
    comparable in scale to the spatial MSE.
    """
    loss = torch.tensor(0.0, device=recon.device)
    for i in range(2):   # block 0 = te, block 1 = ti
        r = recon[:,  i*NX*NY:(i+1)*NX*NY].view(-1, NX, NY)
        t = target[:, i*NX*NY:(i+1)*NX*NY].view(-1, NX, NY)
        t_fft = torch.fft.rfft2(t).abs()
        r_fft = torch.fft.rfft2(r).abs()
        # Normalise by mean GT amplitude so values are ~O(1)
        scale = t_fft.detach().mean(dim=[-2, -1], keepdim=True).clamp(min=1e-8)
        loss = loss + F.mse_loss(r_fft / scale, t_fft / scale)
    return loss / 2


def physics_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Thermal energy conservation loss in log-space.

    E_th = sum_{cells}(n_e T_e + sum_s n_s T_i),  n_e ≈ n_D1  (quasi-neutrality)

    Kinetic energy omitted: E_k = 0.5*m*n*u² loses sign information.
    The gradient d(log E_k)/d(u_n) ∝ u — for u > 0 it pushes u negative
    (sign flip) when E_k_recon < E_k_target, corrupting velocity fields.

    Momentum omitted: net parallel momentum ≈ 0 → p_scale ≈ 0 → diverges.
    """
    te_r, ti_r, na_r, _, _ = denormalize_fields(recon)
    te_t, ti_t, na_t, _, _ = denormalize_fields(target)

    n_e_r = na_r[:, :, :, 1]  # D1 ≈ electron density, (B, NX, NY)
    n_e_t = na_t[:, :, :, 1]
    E_th_r = (n_e_r * te_r + (na_r * ti_r.unsqueeze(-1)).sum(-1)).sum([-1, -2])  # (B,)
    E_th_t = (n_e_t * te_t + (na_t * ti_t.unsqueeze(-1)).sum(-1)).sum([-1, -2])
    return F.mse_loss(torch.log(E_th_r.clamp(min=1e-40)),
                      torch.log(E_th_t.clamp(min=1e-40)))


def loss_function(recon: torch.Tensor, target: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor,
                  beta_current: float) -> torch.Tensor:
    MSE    = F.mse_loss(recon, target, reduction='mean')
    KLD    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    L_spec = spectral_loss(recon, target)
    L_phys = physics_loss(recon, target)
    return MSE + beta_current * KLD + lambda_spectral * L_spec + lambda_physics * L_phys, MSE, KLD


# ---------------------------------------------------------------------------
# Prepare batch
# ---------------------------------------------------------------------------
def prepare_batch(X, te, ti, na, ua, fnixap):
    X, te, ti = X.to(device), te.to(device), ti.to(device)
    na, ua, fnixap = na.to(device), ua.to(device), fnixap.to(device)
    c = normalize_X(X)
    # Clamp to [0,1]: test data can exceed training min/max, making targets
    # outside the sigmoid output range and inflating MSE above 1.0.
    x = normalize_fields(te, ti, na, ua, fnixap).clamp(0.0, 1.0)
    return x, c


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(epoch: int, beta_current: float):
    model.train()
    train_loss = 0.0
    for batch_idx, (X, te, ti, na, ua, fnixap) in enumerate(train_loader):
        x, c = prepare_batch(X, te, ti, na, ua, fnixap)
        optimizer.zero_grad()
        recon, mu, logvar = model(x, c)
        loss, mse, kld = loss_function(recon, x, mu, logvar, beta_current)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  MSE: {:.6f}  KLD: {:.4f}  β: {:.4f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(X), mse.item(), kld.item(), beta_current))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


# ---------------------------------------------------------------------------
# Test / evaluation loop  — production-like: z ~ N(0,I), no encoder at test time
# ---------------------------------------------------------------------------
N_PRIOR_SAMPLES = 10   # number of z draws per test point; more = lower variance

def test(epoch: int) -> float:
    """Evaluate by sampling z from the prior (as in production — no ground truth fields needed).
    Returns mean MSE averaged over N_PRIOR_SAMPLES draws per test sample.
    """
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for X, te, ti, na, ua, fnixap in test_loader:
            x, c = prepare_batch(X, te, ti, na, ua, fnixap)
            B = x.shape[0]
            batch_mse = 0.0
            for _ in range(N_PRIOR_SAMPLES):
                z     = torch.randn(B, latent_size, device=device)
                recon = model.decode(z, c)
                batch_mse += F.mse_loss(recon, x, reduction='sum').item()
            total_mse += batch_mse / N_PRIOR_SAMPLES
    mean_mse = total_mse / (len(test_loader.dataset) * FIELD_SIZE)
    print(f'====> Test prior-MSE (K={N_PRIOR_SAMPLES}): {mean_mse:.6f}')
    return mean_mse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        # Linear beta warmup: 0 → beta over first beta_warmup epochs
        beta_current = beta * min(1.0, epoch / beta_warmup)
        train(epoch, beta_current)
        val_loss = test(epoch)
        scheduler.step()

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'latent_size': latent_size,
                'field_size': FIELD_SIZE,
                'cond_size': COND_SIZE,
            }, 'best_model.pt')
            print(f'  --> Saved best model (epoch {epoch}, loss {val_loss:.4f})')



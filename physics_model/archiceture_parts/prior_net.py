import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_logvar(logvar: torch.Tensor) -> torch.Tensor:
    """Ensure log variance is positive using softplus."""
    return F.softplus(logvar) + 1e-6


class PriorNet(nn.Module):
    """Prior network: Learns p(z|c) from conditioning information."""
    
    def __init__(
        self,
        cond_dim: int,
        latent_dim: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        
        # Hidden layers with residual connection
        self.layer1 = nn.Sequential(
            nn.Linear(self.cond_dim, hidden),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU()
        )
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def fourier_encode(self, c: torch.Tensor) -> torch.Tensor:
        """Apply Fourier positional encoding to conditioning."""
        if not self.use_cond_fourier:
            return c
            
        freqs = self.freqs.to(c.device)
        encoded = [c]
        for freq in freqs:
            encoded.append(torch.sin(freq * c))
            encoded.append(torch.cos(freq * c))
        return torch.cat(encoded, dim=-1)

    def forward(
        self,
        c: torch.Tensor,
        c_emb: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            c: Conditioning input (batch, cond_dim)
            c_emb: Precomputed Fourier embedding (batch, cond_fourier_size), optional
            
        Returns:
            mu: Mean of p(z|c) (batch, latent_dim)
            logvar: Log variance of p(z|c) (batch, latent_dim)
        """
        # Use precomputed embedding or [DELETED]compute Fourier encoding
        if c_emb is not None:
            c_in = c_emb
        else:
            c_in = c
            
        # Forward pass through hidden layers with residual connection
        h1 = self.layer1(c_in)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2) + h1  # residual connection
        
        # Compute prior distribution parameters
        mu = self.fc_mu(h3)
        logvar = softplus_logvar(self.fc_logvar(h3))
        
        return mu, logvar
 
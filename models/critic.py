"""
Preference-conditioned vector critic network.

Takes the flattened observation AND a preference vector λ as input, outputs
per-objective state-value estimates V(s, λ) ∈ ℝ³.

Using a λ-conditioned critic rather than a global EMA baseline eliminates the
mismatch that arises when different preference vectors produce different expected
returns — each (s, λ) pair gets its own baseline.

Architecture:
    [obs (25) ++ λ (3)] → Linear(28→256) → ReLU → Linear(256→256) → ReLU → Linear(256→3)
"""

import torch
import torch.nn as nn


class VectorCritic(nn.Module):
    """
    Parameters
    ----------
    obs_dim : int
        Flattened observation size. highway-env kinematics: 5 vehicles × 5 features = 25.
    lam_dim : int
        Preference vector dimension. 3 for [safety, speed, comfort].
    hidden_dim : int
        Width of the two hidden layers.
    n_objectives : int
        Number of objectives. 3 for [safety, speed, comfort].
    """

    def __init__(
        self,
        obs_dim: int = 25,
        lam_dim: int = 3,
        hidden_dim: int = 256,
        n_objectives: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.lam_dim = lam_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim + lam_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objectives),
        )

    def forward(self, obs: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor [..., obs_dim]   — flattened kinematics, already normalised by env
        lam : Tensor [..., lam_dim]   — preference vector, sums to 1, each entry ≥ 0

        Returns
        -------
        values : Tensor [..., n_objectives]   — [V_safety, V_speed, V_comfort]
        """
        x = torch.cat([obs, lam], dim=-1)
        return self.net(x)

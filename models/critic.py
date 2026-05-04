"""
Preference-conditioned vector critic network.

Takes the flattened observation AND a preference vector λ as input, outputs
per-objective state-value estimates V(s, λ) ∈ ℝ³.

λ is re-concatenated at every layer's input (matching the policy) so the
critic can produce genuinely λ-dependent baselines rather than a near-uniform
value function across preferences.

Architecture:
    [obs (25), λ (3)]    → Linear(28→256)  → ReLU
    [h1, λ (3)]          → Linear(259→256) → ReLU
    [h2, λ (3)]          → Linear(259→3)   → V
"""

import torch
import torch.nn as nn

from models.policy import LAM_SCALE


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

        self.l1 = nn.Linear(obs_dim    + lam_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + lam_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim + lam_dim, n_objectives)

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
        lam_s = lam * LAM_SCALE
        h = torch.relu(self.l1(torch.cat([obs, lam_s], dim=-1)))
        h = torch.relu(self.l2(torch.cat([h,   lam_s], dim=-1)))
        return self.l3(torch.cat([h, lam_s], dim=-1))

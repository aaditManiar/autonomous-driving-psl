"""
Preference-conditioned policy network.

Takes the flattened observation AND a preference vector λ as input, outputs
action logits over the 5 discrete highway-env actions.

Concatenating λ to the observation is equivalent to a shallow hypernetwork:
the network learns to modulate its behaviour based on the preference context.
This is cheaper and more stable than a true PHN that outputs full weight tensors.

Architecture:
    [obs (25) ++ λ (3)] → Linear(28→256) → ReLU → Linear(256→256) → ReLU → Linear(256→5)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ConditionedPolicy(nn.Module):
    """
    Parameters
    ----------
    obs_dim : int
        Flattened observation size. highway-env kinematics: 5 vehicles × 5 features = 25.
    lam_dim : int
        Preference vector dimension. 3 for [safety, speed, comfort].
    hidden_dim : int
        Width of the two hidden layers.
    n_actions : int
        Number of discrete actions. highway-env DiscreteMetaAction: 5.
    """

    def __init__(
        self,
        obs_dim: int = 25,
        lam_dim: int = 3,
        hidden_dim: int = 256,
        n_actions: int = 5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.lam_dim = lam_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim + lam_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor [..., obs_dim]   — flattened kinematics, already normalised by env
        lam : Tensor [..., lam_dim]   — preference vector, sums to 1, each entry ≥ 0

        Returns
        -------
        logits : Tensor [..., n_actions]
        """
        x = torch.cat([obs, lam], dim=-1)
        return self.net(x)

    def act(
        self,
        obs: torch.Tensor,
        lam: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """
        Sample an action and return its log-probability.

        Used during episode collection. The log-probability is kept as a
        computation graph leaf so REINFORCE can differentiate through it.

        Parameters
        ----------
        obs : Tensor [obs_dim]   — single (un-batched) observation
        lam : Tensor [lam_dim]   — single preference vector

        Returns
        -------
        action  : int
        log_prob : Tensor scalar  — log π(action | obs, λ)
        """
        logits = self.forward(obs, lam)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def action_distribution(
        self,
        obs: torch.Tensor,
        lam: torch.Tensor,
    ) -> Categorical:
        """Return the full Categorical distribution (useful for entropy / evaluation)."""
        return Categorical(logits=self.forward(obs, lam))

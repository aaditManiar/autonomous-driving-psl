"""
Preference-conditioned policy network.

Takes the flattened observation AND a preference vector λ as input, outputs
action logits over the 5 discrete highway-env actions.

λ is re-concatenated at every layer's input AND scaled up before injection.
The scaling raises λ's magnitude well above typical normalised obs values so
the linear layers cannot cheaply zero out the λ pathway during training. This
is the structural anti-collapse mechanism for PSL: the network is *forced*
to use λ because it dominates the input.

Architecture:
    [obs (25), LAM_SCALE * λ (3)]    → Linear(28→hidden)  → ReLU
    [h1, LAM_SCALE * λ (3)]          → Linear(hidden+3→hidden) → ReLU
    [h2, LAM_SCALE * λ (3)]          → Linear(hidden+3→n_actions) → logits
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


# λ values arrive in [0, 1] (simplex). Multiply before feeding into the
# network so each component is on the order of typical pre-activation scale,
# making it harder for downstream weights to ignore.
LAM_SCALE = 10.0


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

        self.l1 = nn.Linear(obs_dim    + lam_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + lam_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim + lam_dim, n_actions)

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
        lam_s = lam * LAM_SCALE
        h = torch.relu(self.l1(torch.cat([obs, lam_s], dim=-1)))
        h = torch.relu(self.l2(torch.cat([h,   lam_s], dim=-1)))
        return self.l3(torch.cat([h, lam_s], dim=-1))

    def act(
        self,
        obs: torch.Tensor,
        lam: torch.Tensor,
        return_entropy: bool = False,
    ) -> tuple[int, torch.Tensor] | tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action and return its log-probability.

        Used during episode collection. The log-probability is kept as a
        computation graph leaf so A2C can differentiate through it.

        Parameters
        ----------
        obs : Tensor [obs_dim]   — single (un-batched) observation
        lam : Tensor [lam_dim]   — single preference vector
        return_entropy : bool    — also return H(π(·|obs, λ)) in the same forward pass

        Returns
        -------
        action   : int
        log_prob : Tensor scalar  — log π(action | obs, λ)
        entropy  : Tensor scalar  — H(π(·|obs, λ)), only when return_entropy=True
        """
        logits = self.forward(obs, lam)
        dist = Categorical(logits=logits)
        action = dist.sample()
        if return_entropy:
            return action.item(), dist.log_prob(action), dist.entropy()
        return action.item(), dist.log_prob(action)

    def action_distribution(
        self,
        obs: torch.Tensor,
        lam: torch.Tensor,
    ) -> Categorical:
        """Return the full Categorical distribution (useful for entropy / evaluation)."""
        return Categorical(logits=self.forward(obs, lam))

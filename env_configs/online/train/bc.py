import dataclasses
import torch
import numpy as np
from typing import Dict, Optional, Union
from imitation.data import types
from imitation.util import util
from stable_baselines3.common import policies
from helpers import my_safe_to_tensor



@dataclasses.dataclass(frozen=True)
class BCTrainingMetricsWithMSELoss:
    """Container for the different components of behavior cloning loss."""

    neglogp: torch.Tensor
    entropy: Optional[torch.Tensor]
    ent_loss: torch.Tensor  # set to 0 if entropy is None
    prob_true_act: torch.Tensor
    l2_norm: torch.Tensor
    l2_loss: torch.Tensor
    mse_loss: torch.Tensor
    loss: torch.Tensor

@dataclasses.dataclass(frozen=True)
class BCLossCalculatorWithMSE:
    """Functor to compute the loss used in Behavior Cloning."""

    ent_weight: float
    l2_weight: float
    neglogp_weight: float
    mse_weight: float

    def __call__(
        self,
        policy: policies.ActorCriticPolicy,
        obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, torch.Tensor],
        ],
        acts: Union[torch.Tensor, np.ndarray],
    ) -> BCTrainingMetricsWithMSELoss:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        tensor_obs = types.map_maybe_dict(
            my_safe_to_tensor,
            types.maybe_unwrap_dictobs(obs),
        )
        acts = my_safe_to_tensor(acts)

        # policy.evaluate_actions's type signatures are incorrect.
        # See https://github.com/DLR-RM/stable-baselines3/issues/1679
        (_, log_prob, entropy) = policy.evaluate_actions(
            tensor_obs,  # type: ignore[arg-type]
            acts,
        )
        prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean() if entropy is not None else None

        l2_norms = [torch.sum(torch.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)
        
        acts_pred, _, _ = policy(tensor_obs, deterministic=True)
        mse_loss = torch.nn.functional.mse_loss(acts_pred, acts)


        ent_loss = -self.ent_weight * (entropy if entropy is not None else torch.zeros(1))
        neglogp = -self.neglogp_weight * log_prob
        l2_loss = self.l2_weight * l2_norm
        mse_loss = self.mse_weight * mse_loss
        loss = neglogp + ent_loss + l2_loss + mse_loss
        
        return BCTrainingMetricsWithMSELoss(
            neglogp=neglogp,
            entropy=entropy,
            ent_loss=ent_loss,
            prob_true_act=prob_true_act,
            l2_norm=l2_norm,
            l2_loss=l2_loss,
            mse_loss=mse_loss,
            loss=loss,
        )

"""Policies: abstract base class and concrete implementations."""

import collections
import numpy as np
import torch
import math
from gymnasium import spaces
from torch import nn
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    We consider the possibility of having more dependent dimensions (e.g. time)
    and we sum just the last one (the action dimension)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=-1)
    else:
        tensor = tensor.sum()
    return tensor


def make_proba_distribution(
    action_space: spaces.Space,
    use_sde: bool = False,
    dist_kwargs: Optional[Dict[str, Any]] = None,
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        cls = (
            TimeStateDependentNoiseDistribution
            if use_sde
            else TimeDiagGaussianDistribution
        )
        return cls(get_action_dim(action_space), **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


class IdentityFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space):
        features_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class TimeDiagGaussianDistribution(DiagGaussianDistribution):
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[torch.Tensor]:
        return sum_independent_dims(self.distribution.entropy())


class TimeStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    def entropy(self) -> Optional[torch.Tensor]:
        return sum_independent_dims(self.distribution.entropy())


class ActorExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim

        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in net_arch:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)


class ActorPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy network.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = IdentityFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (
            squash_output and not use_sde
        ), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(
            action_space, use_sde=use_sde, dist_kwargs=dist_kwargs
        )

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.policy_net = self.build_policy_net()
        latent_dim_pi = self.policy_net.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.policy_net: np.sqrt(2),
                self.action_net: 0.01,
            }

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi = self.policy_net.forward(features)

        # Evaluate the values for the given observations
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # TODO: check if this can be safely removed
        # actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, log_prob

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """

        pi_features = super().extract_features(obs, self.features_extractor)
        return pi_features

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi = self.policy_net.forward(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.features_extractor)
        latent_pi = self.policy_net.forward(features)
        return self._get_action_dist_from_latent(latent_pi)

    def build_policy_net(self) -> ActorExtractor:
        """
        Create the policy network.

        :return: The policy network
        """
        return ActorExtractor(
            self.features_dim,
            self.net_arch,
            self.activation_fn,
            device=self.device,
        )


class TransformerActorExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Dict,
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
        max_seq_len=1000,
    ) -> None:
        super().__init__()
        device = get_device(device)

        self.obs_embedding = nn.Linear(feature_dim, net_arch["embed_size"])
        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=net_arch["embed_size"],
                nhead=net_arch["num_heads"],
                dim_feedforward=net_arch["dim_feedforward"],
                activation=activation_fn(),
                dropout=net_arch["dropout"],
                layer_norm_eps=net_arch["layer_norm_eps"],
                batch_first=True,
                norm_first=False,
            ),
            num_layers=net_arch["num_layers"],
        )
        self.latent_dim_pi = net_arch["embed_size"]

        # Positional encoding
        self.max_seq_len = max_seq_len
        # self.positional_encoding = self.create_positional_encoding(
        #     net_arch["embed_size"], max_seq_len
        # )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feature_embedding = self.obs_embedding(features)
        # time_encoded_embeddings = self.time_encode(feature_embedding)

        seq_len = features.size(
            -2
        )  # features are in the format (..., seq_len, feature_dim)
        causal_mask = self.generate_causal_mask(seq_len, features.device)

        return self.transformer_blocks(feature_embedding, mask=causal_mask)

    # TODO: the time is part of the observation, it is likely redundant to add a positional encoding
    # def create_positional_encoding(self, embed_size: int, max_len: int) -> torch.Tensor:
    #     position = torch.arange(0, max_len).unsqueeze(1)  # Shape (max_len, 1)
    #     div_term = torch.exp(
    #         torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
    #     )  # Shape (embed_size // 2)

    #     pos_encoding = torch.zeros(max_len, embed_size)
    #     pos_encoding[:, 0::2] = torch.sin(
    #         position * div_term
    #     )  # Apply sin to even indices
    #     pos_encoding[:, 1::2] = torch.cos(
    #         position * div_term
    #     )  # Apply cos to odd indices
    #     pos_encoding = pos_encoding.unsqueeze(0)  # Shape (1, max_len, embed_size)

    #     return pos_encoding

    # def time_encode(self, feature_embedding: torch.Tensor) -> torch.Tensor:
    #     # Add positional encoding to the embeddings (up to the input sequence length)
    #     seq_len = feature_embedding.size(1)
    #     pos_enc = self.positional_encoding[:, :seq_len, :].to(feature_embedding.device)
    #     return feature_embedding + pos_enc

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Create a causal mask (look-ahead mask)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask


class TransformerActorPolicy(ActorPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[
            BaseFeaturesExtractor
        ] = IdentityFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def build_policy_net(self):
        return TransformerActorExtractor(
            self.features_dim,
            self.net_arch,
            self.activation_fn,
            device=self.device,
        )

    def forward(self, obs, deterministic=False, single_action=False):
        actions, log_prob = super().forward(obs, deterministic)
        if single_action:
            # PyTorch magic to handle the optional batch dimension
            return actions[..., -1, :], log_prob[..., -1]
        else:
            return actions, log_prob

    def predict(
        self,
        observation,
        state,
        episode_start,
        deterministic=False,
        single_action=False,
    ):
        actions, state = super().predict(
            observation, state, episode_start, deterministic
        )
        if single_action:
            # PyTorch magic to handle the optional batch dimension
            return actions[..., -1, :], state
        else:
            return actions, state

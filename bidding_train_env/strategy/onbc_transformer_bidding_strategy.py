import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import json
import torch
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from definitions import ROOT_DIR, ENV_CONFIG_NAME
from online.helpers import load_model, load_vecnormalize
from online.envs.environment_factory import EnvironmentFactory

torch.manual_seed(0)


class ONBCTransformerBiddingStrategy(BaseBiddingStrategy):
    """
    Proximal Policy Optimization (PPO) Strategy
    """

    EPS = 1e-6
    DEFAULT_OBS_KEYS = [
        "time_left",
        "budget_left",
        "budget",
        "cpa",
        "category",
        "last_bid_mean",
        "last_least_winning_cost_mean",
        "last_least_winning_cost_01_pct",
        "last_conversion_mean",
        "last_bid_success",
        "last_cost_mean",
        "last_bid_over_lwc_mean",
        "last_pv_over_lwc_mean",
        "last_pv_over_lwc_90_pct",
        "last_pv_over_lwc_99_pct",
        "current_pvalues_mean",
        "current_pvalues_90_pct",
        "current_pvalues_99_pct",
        "current_pv_num",
    ]

    def __init__(
        self,
        budget=6000,
        name="ONBC-PlayerStrategy",
        cpa=10,
        category=3,
        experiment_path=ROOT_DIR
        / "saved_model"
        / "ONBC"
        / "022_onbc_seed_0_transformer_new_data_realistic_resume_020",
        checkpoint=5100000,
        device="cpu",
        deterministic=True,
        algo="onbc_transformer",
    ):
        super().__init__(budget, name, cpa, category)
        self.device = device
        self.experiment_path = experiment_path
        self.checkpoint = checkpoint
        self.model = load_model(algo, experiment_path, checkpoint)

        train_env_config = json.load(open(experiment_path / ENV_CONFIG_NAME, "r"))
        train_env_config["bids_df_path"] = None
        train_env_config["pvalues_df_path"] = None

        # Train env to create the observation and turn action into bids
        self.train_env = EnvironmentFactory.create(**train_env_config)

        self.vecnormalize = load_vecnormalize(
            experiment_path, checkpoint, self.train_env
        )
        self.vecnormalize.training = False
        self.episode_length = 48
        self.deterministic = deterministic

    def reset(self):
        self.remaining_budget = self.budget
        self.prev_obs = None

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        self.time_step = timeStepIndex
        self.total_budget = self.budget
        self.mean_bid_list = [np.mean(result) for result in historyBid]
        self.mean_pvalues_list = [np.mean(result[:, 0]) for result in historyPValueInfo]
        self.pct_90_pvalues_list = [
            np.percentile(result[:, 0], 90) for result in historyPValueInfo
        ]
        self.pct_99_pvalues_list = [
            np.percentile(result[:, 0], 99) for result in historyPValueInfo
        ]

        self.mean_least_winning_cost_list = [
            np.mean(result) for result in historyLeastWinningCost
        ]
        self.pct_01_least_winning_cost_list = [
            np.percentile(result, 1) for result in historyLeastWinningCost
        ]
        self.mean_conversion_list = [
            np.mean(result[:, 1]) for result in historyImpressionResult
        ]
        self.mean_bid_success_list = [
            np.mean(result[:, 0]) for result in historyAuctionResult
        ]
        self.num_pv_list = [len(result) for result in historyBid]
        self.mean_bid_over_lwc_list = [
            np.mean(bid / (lwc + self.EPS))
            for bid, lwc in zip(historyBid, historyLeastWinningCost)
        ]
        self.mean_pv_over_lwc_list = [
            np.mean(pv[:, 0] / (lwc + self.EPS))
            for pv, lwc in zip(historyPValueInfo, historyLeastWinningCost)
        ]
        self.pct_90_pv_over_lwc_list = [
            np.percentile(pv[:, 0] / (lwc + self.EPS), 90)
            for pv, lwc in zip(historyPValueInfo, historyLeastWinningCost)
        ]
        self.pct_99_pv_over_lwc_list = [
            np.percentile(pv[:, 0] / (lwc + self.EPS), 99)
            for pv, lwc in zip(historyPValueInfo, historyLeastWinningCost)
        ]

        self.mean_cost_list = [np.mean(result[:, 2]) for result in historyAuctionResult]
        self.total_cost_list = [np.sum(result[:, 2]) for result in historyAuctionResult]
        cum_cost = np.cumsum(self.total_cost_list)
        self.budget_left_list = [
            (self.total_budget - cost) / self.total_budget for cost in cum_cost
        ]
        self.time_left_list = list(
            (self.episode_length - np.arange(timeStepIndex + 1)) / self.episode_length
        )

        # Currently unused, but may be useful in the future
        self.bid_slot_list = [result[:, 1] for result in historyAuctionResult]
        self.bid_cost_list = [result[:, 2] for result in historyAuctionResult]
        self.exposure_list = [result[:, 0] for result in historyImpressionResult]
        self.pvalue_sigma_list = [result[:, 1] for result in historyPValueInfo]

        state_dict = self.get_state_dict(pValues)
        state = self.train_env.get_state(state_dict)

        if self.prev_obs is not None:
            assert np.allclose(state[:, :-1], self.prev_obs), (state, self.prev_obs)
        self.prev_obs = state

        obs = self.vecnormalize.normalize_obs(state.T)
        action, _ = self.model.predict(
            obs, deterministic=self.deterministic, single_action=True
        )
        bid_coef, _ = self.train_env.compute_bid_coef(action, pValues, pValueSigmas)
        bids = bid_coef * self.cpa
        return bids

    def get_state_dict(self, pvalues):
        state_dict = {
            "time_left": self.time_left_list,
            "budget_left": [1] + self.budget_left_list,
            "budget": [self.total_budget] * (self.time_step + 1),
            "cpa": [self.cpa] * (self.time_step + 1),
            "category": [self.category] * (self.time_step + 1),
            "last_bid_mean": [0] + self.mean_bid_list,
            "last_least_winning_cost_mean": [0] + self.mean_least_winning_cost_list,
            "last_least_winning_cost_01_pct": [0] + self.pct_01_least_winning_cost_list,
            "last_conversion_mean": [0] + self.mean_conversion_list,
            "last_bid_success": [0] + self.mean_bid_success_list,
            "last_cost_mean": [0] + self.mean_cost_list,
            "last_bid_over_lwc_mean": [0] + self.mean_bid_over_lwc_list,
            "last_pv_over_lwc_mean": [0] + self.mean_pv_over_lwc_list,
            "last_pv_over_lwc_90_pct": [0] + self.pct_90_pv_over_lwc_list,
            "last_pv_over_lwc_99_pct": [0] + self.pct_99_pv_over_lwc_list,
            "current_pvalues_mean": self.mean_pvalues_list + [np.mean(pvalues)],
            "current_pvalues_90_pct": self.pct_90_pvalues_list
            + [np.percentile(pvalues, 90)],
            "current_pvalues_99_pct": self.pct_99_pvalues_list
            + [np.percentile(pvalues, 99)],
            "current_pv_num": self.num_pv_list + [len(pvalues)],
        }
        return state_dict

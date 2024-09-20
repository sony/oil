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


class PpoBiddingStrategy(BaseBiddingStrategy):
    """
    Proximal Policy Optimization (PPO) Strategy
    """
    EPS = 1e-6
    DEFAULT_OBS_KEYS = [
        "time_left",
        "budget_left",
        "historical_bid_mean",
        "last_three_bid_mean",
        "least_winning_cost_mean",
        "pvalues_mean",
        "conversion_mean",
        "bid_success_mean",
        "last_three_least_winning_cost_mean",
        "last_three_pvalues_mean",
        "last_three_conversion_mean",
        "last_three_bid_success_mean",
        "current_pvalues_mean",
        "current_pv_num",
        "last_three_pv_num",
        "pv_num_total",
    ]

    def __init__(
        self,
        budget=6000,
        name="PPO-PlayerStrategy",
        cpa=10,
        category=3,
        experiment_path=ROOT_DIR
        / "saved_model"
        / "ONBC"
        / "017_onbc_seed_0_stoch_exposure_simplified_new_data",
        checkpoint=3700000,
        device="cpu",
        deterministic=True,
        algo="ppo",
    ):
        super().__init__(budget, name, cpa, category)
        self.device = device
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
        self.mean_least_winning_cost_list = [
            np.mean(result) for result in historyLeastWinningCost
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

        # Currently unused, but may be useful in the future
        self.bid_slot_list = [result[:, 1] for result in historyAuctionResult]
        self.bid_cost_list = [result[:, 2] for result in historyAuctionResult]
        self.exposure_list = [result[:, 0] for result in historyImpressionResult]
        self.pvalue_sigma_list = [result[:, 1] for result in historyPValueInfo]

        state_dict = self.get_state_dict(pValues)
        state = self.train_env.get_state(state_dict)
        obs = self.vecnormalize.normalize_obs(state)
        action = self.model.predict(obs, deterministic=self.deterministic)[0]
        bid_coef, _ = self.train_env.compute_bid_coef(action, pValues, pValueSigmas)
        bids = bid_coef * self.cpa
        return bids

    def get_state_dict(self, pvalues):
        if self.time_step == 0:
            return {
                "time_left": 1,
                "budget_left": 1,
                "budget": self.total_budget,
                "cpa": self.cpa,
                "category": self.category,
                "historical_bid_mean": 0,
                "last_three_bid_mean": 0,
                "least_winning_cost_mean": 0,
                "last_least_winning_cost_mean": 0,
                "last_three_least_winning_cost_mean": 0,
                "pvalues_mean": 0,
                "conversion_mean": 0,
                "bid_success_mean": 0,
                "last_pvalues_mean": 0,
                "last_three_pvalues_mean": 0,
                "last_conversion_mean": 0,
                "last_three_conversion_mean": 0,
                "last_bid_success": 0,
                "last_three_bid_success_mean": 0,
                "current_pvalues_mean": np.mean(pvalues),
                "current_pv_num": len(pvalues),
                "last_three_pv_num": 0,
                "pv_num_total": 0,
                "historical_bid_over_lwc_mean": 0,
                "last_bid_over_lwc_mean": 0,
                "last_three_bid_over_lwc_mean": 0,
                "historical_pv_over_lwc_mean": 0,
                "last_pv_over_lwc_mean": 0,
                "last_three_pv_over_lwc_mean": 0,
            }
        else:
            state_dict = {
                "time_left": (self.episode_length - self.time_step)
                / self.episode_length,
                "budget_left": max(self.remaining_budget, 0) / self.total_budget,
                "budget": self.total_budget,
                "cpa": self.cpa,
                "category": self.category,
                "historical_bid_mean": np.mean(self.mean_bid_list),
                "last_three_bid_mean": np.mean(self.mean_bid_list[-3:]),
                "least_winning_cost_mean": np.mean(self.mean_least_winning_cost_list),
                "last_least_winning_cost_mean": self.mean_least_winning_cost_list[-1],
                "last_three_least_winning_cost_mean": np.mean(
                    self.mean_least_winning_cost_list[-3:]
                ),
                "pvalues_mean": np.mean(self.mean_pvalues_list),
                "conversion_mean": np.mean(self.mean_conversion_list),
                "bid_success_mean": np.mean(self.mean_bid_success_list),
                "last_pvalues_mean": self.mean_pvalues_list[-1],
                "last_three_pvalues_mean": np.mean(self.mean_pvalues_list[-3:]),
                "last_conversion_mean": self.mean_conversion_list[-1],
                "last_three_conversion_mean": np.mean(self.mean_conversion_list[-3:]),
                "last_bid_success": self.mean_bid_success_list[-1],
                "last_three_bid_success_mean": np.mean(self.mean_bid_success_list[-3:]),
                "current_pvalues_mean": np.mean(pvalues),
                "current_pv_num": len(pvalues),
                "last_three_pv_num": sum(self.num_pv_list[-3:]),
                "pv_num_total": sum(self.num_pv_list),
                "historical_bid_over_lwc_mean": np.mean(self.mean_bid_over_lwc_list),
                "last_bid_over_lwc_mean": self.mean_bid_over_lwc_list[-1],
                "last_three_bid_over_lwc_mean": np.mean(
                    self.mean_bid_over_lwc_list[-3:]
                ),
                "historical_pv_over_lwc_mean": np.mean(self.mean_pv_over_lwc_list),
                "last_pv_over_lwc_mean": self.mean_pv_over_lwc_list[-1],
                "last_three_pv_over_lwc_mean": np.mean(self.mean_pv_over_lwc_list[-3:]),
            }
        return state_dict

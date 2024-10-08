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
from online.envs.helpers import safe_mean

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

    NO_HISTORY_KEYS = [
        "time_left",
        "budget_left",
        "budget",
        "cpa",
        "category",
        "total_conversions",
        "total_cost",
        "total_cpa",
        "current_pvalues_mean",
        "current_pvalues_90_pct",
        "current_pvalues_99_pct",
        "current_pv_num",
    ]

    HISTORY_KEYS = [
        "least_winning_cost_mean",
        "least_winning_cost_10_pct",
        "least_winning_cost_01_pct",
        "cpa_exceedence_rate",
        "pvalues_mean",
        "conversion_mean",
        "conversion_count",
        "bid_success_mean",
        "successful_bid_position_mean",
        "bid_over_lwc_mean",
        "pv_over_lwc_mean",
        "pv_over_lwc_90_pct",
        "pv_over_lwc_99_pct",
        "pv_num",
        "exposure_count",
        "cost_sum",
    ]

    HISTORY_AND_SLOT_KEYS = [
        "bid_mean",
        "cost_mean",
        "bid_success_count",
        "exposure_mean",
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
        / "026_onbc_seed_0_new_data_realistic_60_obs_resume_023",
        checkpoint=4600000,
        device="cpu",
        deterministic=True,
        algo="ppo",
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
        pvalues_list = [result[:, 0] for result in historyPValueInfo]
        pvalues_sigma_list = [result[:, 1] for result in historyPValueInfo]
        bid_list = historyBid
        bid_success_list = [result[:, 0] > 0 for result in historyAuctionResult]
        bid_position_list = [result[:, 1].astype(int) for result in historyAuctionResult]
        # Here we have to adapt the bid position to the one used in the environment.
        # In the challenge it seems that they use 0 for lost, 1-2-3 for first-second-third slot.
        # In the environment we use -1 for lost, 2-1-0 for first-second-third slot (weird choice sorry)
        bid_position_list = [(3 - result).astype(int) for result in bid_position_list]
        for el in bid_position_list:
            el[el == 3] = -1
        
        bid_cost_list = [result[:, 2] for result in historyAuctionResult]
        bid_exposure_list = [result[:, 0] > 0 for result in historyImpressionResult]
        conversions_list = [result[:, 1] for result in historyImpressionResult]
        least_winning_cost_list = historyLeastWinningCost

        self.time_step = timeStepIndex
        self.total_budget = self.budget
        self.total_conversions = np.sum([np.sum(el) for el in conversions_list])
        self.total_cost = np.sum([np.sum(el) for el in bid_cost_list])
        self.total_cpa = (
            self.total_cost / self.total_conversions
            if self.total_conversions > 0
            else 0
        )
        self.pv_num_total = np.sum([len(el) for el in pvalues_list])

        self.history_info = {
            "least_winning_cost_mean": [np.mean(el) for el in least_winning_cost_list],
            "least_winning_cost_10_pct": [
                np.percentile(el, 10) for el in least_winning_cost_list
            ],
            "least_winning_cost_01_pct": [
                np.percentile(el, 1) for el in least_winning_cost_list
            ],
            "cpa_exceedence_rate": [(self.total_cpa - self.cpa) / self.cpa],
            "pvalues_mean": [np.mean(el) for el in pvalues_list],
            "conversion_mean": [np.mean(el) for el in conversions_list],
            "conversion_count": [np.sum(el) for el in conversions_list],
            "bid_success_mean": [np.mean(el) for el in bid_success_list],
            "successful_bid_position_mean": [
                safe_mean(pos[succ])
                for pos, succ in zip(bid_position_list, bid_success_list)
            ],  # TODO: check bid slot
            "bid_over_lwc_mean": [
                np.mean(bid / (lwc + self.EPS))
                for bid, lwc in zip(bid_list, least_winning_cost_list)
            ],
            "pv_over_lwc_mean": [
                np.mean(pv / (lwc + self.EPS))
                for pv, lwc in zip(pvalues_list, least_winning_cost_list)
            ],
            "pv_over_lwc_90_pct": [
                np.percentile(pv / (lwc + self.EPS), 90)
                for pv, lwc in zip(pvalues_list, least_winning_cost_list)
            ],
            "pv_over_lwc_99_pct": [
                np.percentile(pv / (lwc + self.EPS), 99)
                for pv, lwc in zip(pvalues_list, least_winning_cost_list)
            ],
            "pv_num": [len(el) for el in pvalues_list],
            "exposure_count": [np.sum(el) for el in bid_exposure_list],
            "cost_sum": [np.sum(el) for el in bid_cost_list],
        }
        slot_info_source = {
            "bid_mean": {
                "data": bid_list,
                "func": safe_mean,
                "condition": bid_success_list,
            },
            "cost_mean": {
                "data": bid_cost_list,
                "func": safe_mean,
                "condition": bid_exposure_list,
            },
            "bid_success_count": {
                "data": bid_success_list,
                "func": np.sum,
                "condition": [True] * len(bid_success_list),
            },
            "exposure_mean": {
                "data": bid_exposure_list,
                "func": safe_mean,
                "condition": [True] * len(bid_exposure_list),
            },
        }
        for key, slot_info in slot_info_source.items():
            func = slot_info["func"]
            condition = slot_info["condition"]
            data = slot_info["data"]
            self.history_info[key] = [
                func(el) for el in data
            ]  # For some reason we did not apply the condition for the non-slot-specific metric
            for slot in range(3):
                self.history_info[f"{key}_slot_{3 - slot}"] = [
                    func(el[np.logical_and(pos == slot, cond)])
                    for el, pos, cond in zip(data, bid_position_list, condition)
                ]

        state_dict = self.get_state_dict(pValues)
        state = self.train_env.get_state(state_dict)
        obs = self.vecnormalize.normalize_obs(state)
        action = self.model.predict(obs, deterministic=self.deterministic)[0]
        bid_coef, _ = self.train_env.compute_bid_coef(action, pValues, pValueSigmas)
        bids = bid_coef * self.cpa
        return bids

    def get_state_dict(self, pvalues):
        state_dict = {
            "time_left": (self.episode_length - self.time_step) / self.episode_length,
            "budget_left": max(self.remaining_budget, 0) / self.total_budget,
            "budget": self.total_budget,
            "cpa": self.cpa,
            "category": self.category,
            "total_conversions": self.total_conversions,
            "total_cost": self.total_cost,
            "total_cpa": self.total_cpa,
            "pv_num_total": self.pv_num_total,
            "current_pvalues_mean": np.mean(pvalues),
            "current_pvalues_90_pct": np.percentile(pvalues, 90),
            "current_pvalues_99_pct": np.percentile(pvalues, 99),
            "current_pv_num": len(pvalues),
        }
        for key, info in self.history_info.items():
            state_dict[f"last_{key}"] = safe_mean(info[-1:])
            state_dict[f"last_three_{key}"] = safe_mean(info[-3:])
            state_dict[f"historical_{key}"] = safe_mean(info)

        # Correction for backward compatibility
        state_dict["last_three_pv_num"] = np.sum(self.history_info["pv_num"][-3:])

        # Deprecated keys compatibility
        state_dict["least_winning_cost_mean"] = state_dict[
            "historical_least_winning_cost_mean"
        ]
        state_dict["least_winning_cost_10_pct"] = state_dict[
            "historical_least_winning_cost_10_pct"
        ]
        state_dict["least_winning_cost_01_pct"] = state_dict[
            "historical_least_winning_cost_01_pct"
        ]
        state_dict["pvalues_mean"] = state_dict["historical_pvalues_mean"]
        state_dict["conversion_mean"] = state_dict["historical_conversion_mean"]
        state_dict["bid_success_mean"] = state_dict["historical_bid_success_mean"]
        state_dict["last_bid_success"] = state_dict["last_bid_success_mean"]
        state_dict["historical_cost_slot_1_mean"] = state_dict[
            "historical_cost_mean_slot_1"
        ]
        state_dict["historical_cost_slot_2_mean"] = state_dict[
            "historical_cost_mean_slot_2"
        ]
        state_dict["historical_cost_slot_3_mean"] = state_dict[
            "historical_cost_mean_slot_3"
        ]
        state_dict["last_cost_slot_1_mean"] = state_dict["last_cost_mean_slot_1"]
        state_dict["last_cost_slot_2_mean"] = state_dict["last_cost_mean_slot_2"]
        state_dict["last_cost_slot_3_mean"] = state_dict["last_cost_mean_slot_3"]
        state_dict["last_three_cost_slot_1_mean"] = state_dict[
            "last_three_cost_mean_slot_1"
        ]
        state_dict["last_three_cost_slot_2_mean"] = state_dict[
            "last_three_cost_mean_slot_2"
        ]
        state_dict["last_three_cost_slot_3_mean"] = state_dict[
            "last_three_cost_mean_slot_3"
        ]
        return state_dict

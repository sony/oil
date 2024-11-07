import gymnasium as gym
import numpy as np
import pandas as pd
from .helpers import safe_mean, safe_max
from definitions import (
    DEFAULT_ACT_KEYS,
    DEFAULT_OBS_KEYS,
    DEFAULT_RWD_WEIGHTS,
    HISTORY_KEYS,
    NO_HISTORY_KEYS,
    HISTORY_AND_SLOT_KEYS,
)


class BiddingEnv(gym.Env):
    EPS = 1e-6

    def __init__(
        self,
        pvalues_df_path=None,
        bids_df_path=None,
        budget_range=(6000, 6000),
        target_cpa_range=(8, 8),
        advertiser_id=None,
        obs_keys=DEFAULT_OBS_KEYS,
        rwd_weights=DEFAULT_RWD_WEIGHTS,
        act_keys=DEFAULT_ACT_KEYS,
        auction_noise=(0, 0),
        stochastic_exposure=False,
        deterministic_conversion=False,
        flex_oracle=False,
        flex_oracle_cost_weight=0.5,  # How to mix lower and upper cost
        exclude_self_bids=True,
        two_slopes_action=False,
        detailed_bid=False,
        batch_state=False,  # For evaluation, return matrix obs
        advertiser_categories=None,
        seed=0,
    ):
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(obs_keys) + 2 * detailed_bid,),
            dtype=np.float32,
        )
        self.act_keys = act_keys
        self.pvalues_key_pos = self.act_keys.index("pvalue")
        low_lim = -10
        high_lim = 10
        if two_slopes_action:
            self.action_space = gym.spaces.Box(
                low=low_lim, high=high_lim, shape=(3,), dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=-10, high=10, shape=(len(self.act_keys),), dtype=np.float32
            )
        self.obs_keys = obs_keys
        self.rwd_weights = rwd_weights
        if isinstance(auction_noise, (int, float)):
            self.auction_noise_min = self.auction_noise_max = auction_noise
        else:
            self.auction_noise_min, self.auction_noise_max = auction_noise
        self.stochastic_exposure = stochastic_exposure
        self.deterministic_conversion = deterministic_conversion
        self.flex_oracle = flex_oracle
        self.exclude_self_bids = exclude_self_bids
        self.two_slopes_action = two_slopes_action
        self.cost_weight = flex_oracle_cost_weight
        self.detailed_bid = detailed_bid
        self.batch_state = batch_state

        if pvalues_df_path is None or bids_df_path is None:
            print("Warning: creating a dummy environment with no dataset")
        else:
            self.pvalues_df = self.load_pvalues_df(pvalues_df_path)
            self.bids_df_list = self.load_bids_df(bids_df_path)
            self.episode_length = len(self.bids_df_list[0].timeStepIndex.unique())
            self.advertiser_categories = advertiser_categories

            categories_df = self.pvalues_df.groupby(
                "advertiserNumber"
            ).advertiserCategoryIndex.first()
            self.advertiser_category_dict = {
                advertiser: category
                for advertiser, category in zip(
                    categories_df.index, categories_df.values
                )
            }
            if advertiser_id is None:
                self.advertiser_list = list(self.pvalues_df.advertiserNumber.unique())
                if advertiser_categories is not None:
                    self.advertiser_list = [
                        ad
                        for ad in self.advertiser_list
                        if self.advertiser_category_dict[ad] in advertiser_categories
                    ]
            else:
                self.advertiser_list = [advertiser_id]
            self.period_list = list(self.pvalues_df.deliveryPeriodIndex.unique())
            self.budget_range = budget_range
            self.target_cpa_range = target_cpa_range
            self.reset(seed=seed)

    def reset_campaign_params(
        self, budget=None, target_cpa=None, advertiser=None, period=None
    ):
        self.advertiser = self.sample_advertiser() if advertiser is None else advertiser
        self.period = self.sample_period() if period is None else period
        self.total_budget = self.sample_budget() if budget is None else budget
        self.target_cpa = self.sample_cpa() if target_cpa is None else target_cpa
        self.time_step = 0
        self.remaining_budget = self.total_budget
        self.total_conversions = 0
        self.total_cost = 0
        self.total_cpa = 0
        self.pv_num_total = 0
        self.history_info = {key: [] for key in HISTORY_KEYS}
        self.history_info.update({key: [] for key in NO_HISTORY_KEYS})
        self.history_info.update({key: [] for key in HISTORY_AND_SLOT_KEYS})
        self.history_info.update(
            {
                f"{key}_slot_{slot}": []
                for key in HISTORY_AND_SLOT_KEYS
                for slot in range(1, 4)
            }
        )

        self.episode_bids_df = self.get_episode_bids_df()
        self.episode_pvalues_df = self.get_episode_pvalues_df()

        # TODO: try to merge some of these attributes
        self.ranked_df = None
        self.impression_ids = None
        self.slots = None
        self.pv_costs = None
        self.time_steps_arr = None
        self.eff_pv_table = None
        self.eff_cost_table = None
        self.impression_ids_f = None
        self.slots_f = None
        self.pv_costs_f = None
        self.time_steps_arr_f = None
        self.eff_pv_table_f = None
        self.eff_cost_table_f = None
        self.pvs = None
        self.costs = None
        self.eff_pvs_with_up = None
        self.eff_costs_with_up = None
        self.imp_idx_arr = None
        self.pv_idx = 0
        self.cur_pv_num = None
        self.cur_matrix_state = None
        self.cur_action = None
        self.cur_oracle_action = None

    def reset(
        self,
        budget=None,
        target_cpa=None,
        advertiser=None,
        period=None,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)
        self.reset_campaign_params(budget, target_cpa, advertiser, period)
        pvalues, pvalues_std = self.get_pvalues_mean_and_std()
        state_dict = self.get_state_dict(pvalues, pvalues_std)
        state = self.get_state(state_dict)
        if self.detailed_bid and not self.batch_state:
            self.cur_matrix_state = state
            self.cur_pv_num = len(pvalues)
            self.cur_action = np.zeros((self.cur_pv_num, len(self.act_keys)))
            self.cur_oracle_action = np.zeros((self.cur_pv_num, len(self.act_keys)))
            state = self.cur_matrix_state[self.pv_idx]
        return state, {}

    def step(self, action):
        if self.detailed_bid and not self.batch_state:
            # The action is just for one pv, we need to keep track of the bids for all pvs
            self.cur_action[self.pv_idx, self.pvalues_key_pos] = action
            self.pv_idx = (self.pv_idx + 1) % self.cur_pv_num
            if self.pv_idx == 0:
                # We have collected actions for each pv, execute the step
                obs, reward, terminated, truncated, info = self.execute_step(
                    self.cur_action
                )
                self.cur_pv_num = obs.shape[0]
                self.cur_action = np.zeros((self.cur_pv_num, len(self.act_keys)))
                self.cur_matrix_state = obs
            else:
                reward = 0
                terminated = False
                truncated = False
                info = {}
            state = self.cur_matrix_state[self.pv_idx]
            return state, reward, terminated, truncated, info
        else:
            return self.execute_step(action)

    def execute_step(self, action):
        # Get current pvalues to compute the bids
        pvalues, pvalues_sigma = self.get_pvalues_mean_and_std()
        bid_coef = self.compute_bid_coef(action, pvalues, pvalues_sigma)
        advertiser_bids = bid_coef * self.target_cpa
        return self.place_bids(advertiser_bids, pvalues, pvalues_sigma)

    def place_bids(self, advertiser_bids, pvalues, pvalues_sigma):
        advertiser_bids = np.clip(advertiser_bids, 0, self.max_bid)
        bid_data = self.get_bid_data()
        top_bids = bid_data.bid.item()
        top_bids_cost = bid_data.cost.item()
        least_winning_cost = top_bids_cost[:, 0]

        top_bids_exposed = bid_data.isExposed.item()
        if self.stochastic_exposure:
            # We simulate exposure as a Bernoulli with the mean exposure probability per slot
            # It could be unrealistic that one bid is not exposed but a lower one is,
            # but it should be irrelevant from a single agent's perspective
            exposure_prob_per_slot = np.mean(top_bids_exposed, axis=0)
            top_bids_exposed = self.np_random.binomial(
                n=1, p=exposure_prob_per_slot, size=top_bids_exposed.shape
            )

        bid_success, bid_position, bid_exposed, bid_cost, bid_conversion = (
            self.simulate_ad_bidding(
                pvalues,
                pvalues_sigma,
                advertiser_bids,
                top_bids,
                top_bids_exposed,
                least_winning_cost,
            )
        )
        self.time_step += 1
        self.total_conversions += np.sum(bid_conversion)
        self.total_cost += np.sum(bid_cost)
        self.total_cpa = (
            self.total_cost / self.total_conversions
            if self.total_conversions > 0
            else 0
        )
        self.pv_num_total += len(pvalues)
        self.remaining_budget -= np.sum(bid_cost)
        terminated = self.time_step >= self.episode_length
        dense_reward = self.compute_score(np.sum(bid_cost), np.sum(bid_conversion))

        # Update the history which is used to compute the observations
        history_info_update = {
            "least_winning_cost_mean": np.mean(least_winning_cost),
            "least_winning_cost_10_pct": np.percentile(least_winning_cost, 10),
            "least_winning_cost_01_pct": np.percentile(least_winning_cost, 1),
            "cpa_exceedence_rate": (self.total_cpa - self.target_cpa) / self.target_cpa,
            "pvalues_mean": np.mean(pvalues),
            "conversion_mean": np.mean(bid_conversion),
            "conversion_count": np.sum(bid_conversion),
            "bid_success_mean": np.mean(bid_success),
            "successful_bid_position_mean": safe_mean(bid_position[bid_success]),
            "bid_over_lwc_mean": np.mean(
                advertiser_bids / (least_winning_cost + self.EPS)
            ),
            "pv_over_lwc_mean": np.mean(pvalues / (least_winning_cost + self.EPS)),
            "pv_over_lwc_90_pct": np.percentile(
                pvalues / (least_winning_cost + self.EPS), 90
            ),
            "pv_over_lwc_99_pct": np.percentile(
                pvalues / (least_winning_cost + self.EPS), 99
            ),
            "pv_num": len(pvalues),
            "exposure_count": np.sum(bid_exposed),
            "cost_sum": np.sum(bid_cost),
        }
        slot_info_source = {
            "bid_mean": {
                "data": advertiser_bids,
                "func": safe_mean,
                "condition": bid_success,
            },
            "cost_mean": {
                "data": bid_cost,
                "func": safe_mean,
                "condition": bid_exposed,
            },
            "bid_success_count": {
                "data": bid_success,
                "func": np.sum,
                "condition": True,
            },
            "exposure_mean": {
                "data": bid_exposed,
                "func": safe_mean,
                "condition": True,
            },
        }
        for key, slot_info in slot_info_source.items():
            func = slot_info["func"]
            condition = slot_info["condition"]
            data = slot_info["data"]
            history_info_update[key] = func(data)
            for slot in range(3):
                history_info_update[f"{key}_slot_{3 - slot}"] = func(
                    data[np.logical_and(bid_position == slot, condition)]
                )

        for key, value in history_info_update.items():
            self.history_info[key].append(value)

        info = {
            "action": np.sum(advertiser_bids)
            / (np.sum(pvalues) + self.EPS)
            / self.target_cpa,
            "bid": np.mean(advertiser_bids),
        }
        if terminated:
            cpa = (
                self.total_cost / self.total_conversions + self.EPS
                if self.total_conversions > 0
                else 0
            )
            sparse_reward = self.compute_score(self.total_cost, self.total_conversions)
            reward_dict = {
                "sparse": sparse_reward,
                "dense": dense_reward,
            }
            episode_end_info = {
                "conversions": self.total_conversions,
                "cost": self.total_cost,
                "cpa": cpa,
                "target_cpa": self.target_cpa,
                "budget": self.total_budget,
                "avg_pvalues": np.mean(self.history_info["pvalues_mean"]),
                "score_over_pvalue": sparse_reward
                / np.mean(self.history_info["pvalues_mean"]),
                "score_over_budget": sparse_reward / self.total_budget,
                "score_over_cpa": sparse_reward / cpa if cpa > 0 else 0,
                "cost_over_budget": self.total_cost / self.total_budget,
                "target_cpa_over_cpa": self.target_cpa / cpa if cpa > 0 else 0,
                "score": sparse_reward,
            }
            info.update(episode_end_info)
            info.update(reward_dict)
            reward = self.compute_reward(reward_dict)

            # There are no new pvalues as the episode is over, just pass the old ones (anyway unused)
            new_pvalues, new_pvalues_std = pvalues, pvalues_sigma
        else:
            reward_dict = {"sparse": 0, "dense": dense_reward}
            info.update(reward_dict)
            reward = self.compute_reward(reward_dict)

            # Get the new pvalues for the next state
            new_pvalues, new_pvalues_std = self.get_pvalues_mean_and_std()
        state_dict = self.get_state_dict(new_pvalues, new_pvalues_std)
        state = self.get_state(state_dict)
        return state, reward, terminated, False, info

    def compute_score(self, cost, conversions):
        cpa = cost / conversions if conversions > 0 else 0.0
        cpa_coeff = min(1, (self.target_cpa / cpa) ** 2) if cpa > 0 else 0.0
        score = cpa_coeff * conversions
        return score

    def compute_reward(self, reward_dict):
        reward = 0
        for key, weight in self.rwd_weights.items():
            reward += weight * reward_dict[key]
        return reward

    def simulate_ad_bidding(
        self,
        pvalues: np.ndarray,
        pvalues_sigma: np.ndarray,
        advertiser_bids: np.ndarray,
        top_bids: np.ndarray,
        top_bids_exposed: np.ndarray,
        least_winning_cost: np.ndarray,
    ):
        bid_success, bid_position, bid_exposed, bid_cost = (
            self.compute_success_exposition_cost(
                advertiser_bids, top_bids, top_bids_exposed, least_winning_cost
            )
        )

        bid_success, bid_position, bid_exposed, bid_cost = self.handle_overcost(
            bid_success,
            bid_position,
            bid_exposed,
            bid_cost,
            advertiser_bids,
            top_bids,
            top_bids_exposed,
            least_winning_cost,
        )

        if self.deterministic_conversion:
            bid_conversion = pvalues * bid_exposed
        else:
            pvalues_sampled = np.clip(
                self.np_random.normal(pvalues, pvalues_sigma), 0, 1
            )
            bid_conversion = (
                self.np_random.binomial(n=1, p=pvalues_sampled) * bid_exposed
            )
        return bid_success, bid_position, bid_exposed, bid_cost, bid_conversion

    def compute_success_exposition_cost(
        self, advertiser_bids, top_bids, top_bids_exposed, least_winning_cost
    ):
        advertiser_bid_higher = advertiser_bids[:, None] >= top_bids
        bid_success = advertiser_bid_higher.any(axis=1)
        bid_position = np.sum(advertiser_bid_higher, axis=1) - 1

        # Exposed is 0 if the bid is not successful
        bid_exposed = np.zeros_like(bid_position)
        bid_exposed[bid_success] = top_bids_exposed[
            bid_success, bid_position[bid_success]
        ]

        # If I am higher than a bid, I pay that bid's price. No payment for not winning or not exposing
        bid_cost = top_bids[np.arange(len(bid_position)), bid_position] * bid_exposed
        return bid_success, bid_position, bid_exposed, bid_cost

    def handle_overcost(
        self,
        bid_success,
        bid_position,
        bid_exposed,
        bid_cost,
        advertiser_bids,
        top_bids,
        top_bids_exposed,
        least_winning_cost,
    ):
        total_cost = sum(bid_cost)
        # Remove random bids until below the budget
        while total_cost > self.remaining_budget:
            # Set a fraction of successful bids to 0
            over_cost_ratio = (total_cost - self.remaining_budget) / total_cost
            bid_success_index = np.where(bid_success)[0]
            dropped_index = self.np_random.choice(
                bid_success_index,
                int(np.ceil(bid_success_index.shape[0] * over_cost_ratio)),
                replace=False,
            )
            advertiser_bids[dropped_index] = 0
            bid_success, bid_position, bid_exposed, bid_cost = (
                self.compute_success_exposition_cost(
                    advertiser_bids, top_bids, top_bids_exposed, least_winning_cost
                )
            )
            total_cost = sum(bid_cost)
        return bid_success, bid_position, bid_exposed, bid_cost

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def sample_budget(self):
        return self.np_random.uniform(*self.budget_range)

    def sample_cpa(self):
        return self.np_random.uniform(*self.target_cpa_range)

    def sample_advertiser(self):
        return self.np_random.choice(self.advertiser_list)

    def sample_period(self):
        return self.np_random.choice(self.period_list)

    def get_state_dict(self, pvalues, pvalues_std):
        state_dict = {
            "time_left": (self.episode_length - self.time_step) / self.episode_length,
            "budget_left": max(self.remaining_budget, 0) / self.total_budget,
            "budget": self.total_budget,
            "cpa": self.target_cpa,
            "category": self.advertiser_category_dict[self.advertiser],
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

        # For matrix obs
        state_dict["pvalues"] = pvalues
        state_dict["pvalues_std"] = pvalues_std

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

    def get_state(self, state_dict):
        state_vec = np.array([state_dict[key] for key in self.obs_keys]).astype(
            np.float32
        )
        if self.detailed_bid:
            # Create a matrix with pv, pv_sigma and state for each impression
            state = np.zeros((state_dict["current_pv_num"], 2 + len(self.obs_keys)))
            state[:, 0] = state_dict["pvalues"]
            state[:, 1] = state_dict["pvalues_std"]
            state[:, 2:] = state_vec
        else:
            state = state_vec
        return state

    def get_bid_basis_dict(self, pvalues, pvalues_sigma):
        return {
            "pvalue": pvalues,
            "pvalue_sigma": pvalues_sigma,
            "pvalue_square": 1e2 * pvalues**2,
            "pvalue_sigma_square": 1e2 * pvalues_sigma**2,
            "pvalue_sigma_pvalue": 1e2 * pvalues_sigma * pvalues,
            "pvalue_sqrt": 1e-2 * np.sqrt(pvalues),
        }

    def get_bid_basis(self, pvalues, pvalues_sigma):
        basis_dict = self.get_bid_basis_dict(pvalues, pvalues_sigma)
        return np.stack([basis_dict[key] for key in self.act_keys], axis=1)

    def compute_bid_coef(self, action, pvalues, pvalues_sigma):
        if self.two_slopes_action:
            log_y_0, x_0, slope = action
            x_0 = x_0 * 1e-3
            slope = slope * 1e3
            y_0 = np.exp(log_y_0)
            y_pred = y_0 * np.ones_like(pvalues)
            y_pred[pvalues >= x_0] = y_0 + slope * (pvalues[pvalues >= x_0] - x_0)

            action = np.zeros((len(pvalues), len(self.act_keys)))
            action[:, self.pvalues_key_pos] = 1 / y_pred
        else:
            action = np.atleast_2d(action).copy()
            action[:, self.pvalues_key_pos] = np.exp(action[:, self.pvalues_key_pos])
        bid_basis = self.get_bid_basis(pvalues, pvalues_sigma)
        bid_coef = np.clip(np.einsum("nk,nk->n", action, bid_basis), 0, np.inf)
        return bid_coef

    def get_pvalues_mean_and_std(self):
        p_row = self.episode_pvalues_df[
            self.episode_pvalues_df.timeStepIndex == self.time_step
        ]
        return p_row.pValue.item(), p_row.pValueSigma.item()

    def get_bid_data(self):
        bid_data = self.episode_bids_df[
            (self.episode_bids_df.timeStepIndex == self.time_step)
        ]
        return bid_data

    def load_pvalues_df(self, pvalues_df_path):
        print(f"Loading pvalues from {pvalues_df_path}")
        pvalues_df = pd.read_parquet(pvalues_df_path)
        pvalues_df["pValue"] = pvalues_df["pValue"].apply(np.array)
        pvalues_df["pValueSigma"] = pvalues_df["pValueSigma"].apply(np.array)
        return pvalues_df

    def load_bids_df(self, bids_df_path):
        if not isinstance(bids_df_path, list):
            bids_df_path = [bids_df_path]
        bids_df_list = []
        for path in bids_df_path:
            print(f"Loading bids from {path}")
            bids_df = pd.read_parquet(path)
            bids_df["bid"] = bids_df["bid"].apply(np.stack)
            bids_df["isExposed"] = bids_df["isExposed"].apply(np.stack)
            bids_df["cost"] = bids_df["cost"].apply(np.stack)
            bids_df["deliveryPeriodIndex"] = bids_df["deliveryPeriodIndex"].apply(
                lambda x: x % 28
            )
            if self.exclude_self_bids:
                bids_df["advertiserNumber"] = bids_df["advertiserNumber"].apply(
                    np.stack
                )
            bids_df_list.append(bids_df)
        return bids_df_list

    def get_baseline_action(self):
        remaining_budget_excess = (
            self.remaining_budget
            * self.episode_length
            / (self.total_budget * (self.episode_length - self.time_step + 1))
        )
        return 0.8 * remaining_budget_excess

    def noisy_bid_and_cost(self, row):
        bid = row["bid"]
        cost = row["cost"]
        second_price_ratio = cost[:, 0] / bid[:, 0]

        # Add noise to bids
        noisy_bid = bid * (
            self.np_random.uniform(
                self.auction_noise_min, self.auction_noise_max, bid.shape
            )
            + 1
        )
        noisy_bid = np.sort(noisy_bid, axis=1)
        noisy_cost = np.zeros_like(noisy_bid)
        noisy_cost[:, 0] = noisy_bid[:, 0] * second_price_ratio
        noisy_cost[:, 1:] = noisy_bid[:, :-1]

        # Return the modified noisy_bid and noisy_cost
        return pd.Series([noisy_bid, noisy_cost])

    @staticmethod
    def update_bids_excluding_self(row, advertiser):
        # Extract the bid, advertiser, and cost arrays from the row
        top_bids = row["bid"]
        top_bids_advertiser = row["advertiserNumber"]
        top_bids_cost = row["cost"]
        full_top_bids = np.concatenate((top_bids_cost[:, :1], top_bids), axis=1)

        # Mask for valid (non-matching advertiser) and the advertiser to replace
        ad_id_mask = top_bids_advertiser == advertiser
        found_mask = np.any(ad_id_mask, axis=1).reshape(-1, 1)
        full_mask = np.concatenate((~found_mask, ad_id_mask), axis=1)

        # Update the bids: removing the self.advertiser and keeping the next bid
        updated_top_bids = full_top_bids[~full_mask].reshape(-1, 3)

        return updated_top_bids

    def get_episode_bids_df(self):
        random_idx = self.np_random.choice(len(self.bids_df_list))
        bids_df = self.bids_df_list[random_idx]
        ep_bids_df = bids_df[bids_df.deliveryPeriodIndex == self.period].copy()

        if self.exclude_self_bids:
            ep_bids_df["bid"] = ep_bids_df.apply(
                lambda row: self.update_bids_excluding_self(row, self.advertiser),
                axis=1,
            )

        if self.auction_noise_min != 0 or self.auction_noise_max != 0:
            ep_bids_df[["bid", "cost"]] = ep_bids_df.apply(
                lambda x: self.noisy_bid_and_cost(x), axis=1
            )
        return ep_bids_df

    def get_episode_pvalues_df(self):
        ep_pv_df = self.pvalues_df[
            (self.pvalues_df.advertiserNumber == self.advertiser)
            & (self.pvalues_df.deliveryPeriodIndex == self.period)
        ].copy()
        return ep_pv_df

    def compute_ranked_impressions_df(self):
        # Extract pvalues and costs as lists of arrays
        pvalues_list = np.concatenate(self.episode_pvalues_df.pValue.values)
        min_cost_list = np.concatenate(
            self.episode_bids_df.cost.apply(lambda x: x[:, 0]).values
        )
        df = pd.DataFrame(
            {
                "time_step": np.repeat(
                    np.arange(len(self.episode_pvalues_df)),
                    self.episode_pvalues_df.pValue.apply(len),
                ),
                "ad_impression_id": np.concatenate(
                    [np.arange(len(pv)) for pv in self.episode_pvalues_df.pValue]
                ),
                "cost": min_cost_list,
                "pvalue": pvalues_list,
            }
        )
        df["pv_over_cost"] = df["pvalue"] / (df["cost"] + self.EPS)

        # Sort once and store the result for later use
        df_sorted = df.sort_values(by="pv_over_cost", ascending=False).reset_index(
            drop=True
        )
        return df_sorted

    def get_oracle_action(self):
        if self.detailed_bid and not self.batch_state:
            if self.pv_idx == 0:
                self.cur_oracle_action = self.get_action_from_correct_oracle()
            if self.flex_oracle:
                return self.cur_oracle_action[self.pv_idx]
            else:
                return self.cur_oracle_action
        else:
            return self.get_action_from_correct_oracle()

    def get_action_from_correct_oracle(self):
        if self.flex_oracle:
            return self.get_flex_oracle_action()
        else:
            return self.get_realistic_oracle_action()

    def get_realistic_oracle_action(self):
        if self.impression_ids is None:
            # Sort the impression opportunities for all slots
            self.cost_table = np.vstack(self.episode_bids_df.bid)
            exposed_table = np.vstack(self.episode_bids_df.isExposed)
            pvalues_arr = np.concatenate(self.episode_pvalues_df.pValue.to_list())
            time_arr = np.repeat(
                np.arange(len(self.episode_pvalues_df)),
                self.episode_pvalues_df.pValue.apply(len),
            )
            exposed_prob = np.mean(exposed_table, axis=0)
            self.eff_cost_table = self.cost_table * exposed_prob
            self.eff_pv_table = np.outer(pvalues_arr, exposed_prob)
            pv_cost_table = self.eff_pv_table / self.eff_cost_table

            n_impressions, n_slots = pv_cost_table.shape

            # Generate impression ids and slots using vectorization
            self.impression_ids = np.repeat(np.arange(n_impressions), n_slots)
            self.slots = np.tile(np.arange(n_slots), n_impressions)

            # Assign pv_costs directly by flattening the pv_cost_table
            self.pv_costs = pv_cost_table.flatten()

            # Repeat time_arr for all slots
            self.time_steps_arr = np.repeat(time_arr, n_slots)

            # Sort by pv/cost (descending)
            sort_indices = np.argsort(self.pv_costs)[::-1]
            self.impression_ids = self.impression_ids[sort_indices]
            self.slots = self.slots[sort_indices]
            self.pv_costs = self.pv_costs[sort_indices]
            self.time_steps_arr = self.time_steps_arr[sort_indices]
        else:
            n_impressions, n_slots = self.eff_pv_table.shape
            valid_indices = self.time_steps_arr >= self.time_step
            self.impression_ids = self.impression_ids[valid_indices]
            self.slots = self.slots[valid_indices]
            self.pv_costs = self.pv_costs[valid_indices]
            self.time_steps_arr = self.time_steps_arr[valid_indices]

        if self.impression_ids.size == 0:
            oracle_action = 1
        else:
            # Variables to track total cost, total pv, best score, and corresponding alpha
            cum_cost = self.total_cost
            cum_pv = self.total_conversions
            best_score = -np.inf

            # Store current slot selection per impression
            current_slot = [
                -1
            ] * n_impressions  # -1 means no slot is currently selected

            # Step 3: Iterate through the sorted impressions
            for imp_id, new_slot, pv_cost in zip(
                self.impression_ids, self.slots, self.pv_costs
            ):

                # Remove the old contribution from the previous slot if it was set
                if current_slot[imp_id] != -1:
                    prev_slot = current_slot[imp_id]
                    cum_cost -= self.eff_cost_table[imp_id, prev_slot]
                    cum_pv -= self.eff_pv_table[imp_id, prev_slot]

                # Add the new contribution for the current slot
                cum_cost += self.eff_cost_table[imp_id, new_slot]
                cum_pv += self.eff_pv_table[imp_id, new_slot]

                # Update the current slot for this impression
                current_slot[imp_id] = new_slot

                # Early termination if cost exceeds the budget
                if cum_cost > self.total_budget:
                    break

                # Compute CPA and score
                cpa = cum_cost / cum_pv if cum_pv > 0 else np.inf
                score = cum_pv * min((self.target_cpa / cpa) ** 2, 1)

                # Find the maximum score within the budget constraint
                if score > best_score:
                    best_score = score
                    best_pv_cost = pv_cost  # Set alpha as cost / pv of max score

            # Transform the best pv over cost into the action
            if best_score < 0:
                # We cannot improve the score, just output 1
                action = 0
            else:
                action = -np.log(best_pv_cost * self.target_cpa)
                
        if self.two_slopes_action:
            oracle_action = np.array([1 / action, 0, 0])
        else:
            oracle_action = np.zeros(len(self.act_keys))
            oracle_action[self.pvalues_key_pos] = action
        return oracle_action

    def get_flex_oracle_action(self):
        if self.impression_ids_f is None:
            # Sort the impression opportunities for all slots
            self.cost_table = np.vstack(self.episode_bids_df.bid)
            exposed_table = np.vstack(self.episode_bids_df.isExposed)
            pvalues_arr = np.concatenate(self.episode_pvalues_df.pValue.to_list())
            self.pv_table = np.outer(pvalues_arr, np.ones(3))
            time_arr = np.repeat(
                np.arange(len(self.episode_pvalues_df)),
                self.episode_pvalues_df.pValue.apply(len),
            )
            impression_indices = np.concatenate(
                [np.arange(len(pv)) for pv in self.episode_pvalues_df.pValue]
            )
            exposed_prob = np.mean(exposed_table, axis=0)
            self.eff_cost_table_f = self.cost_table * exposed_prob
            self.eff_pv_table_f = np.outer(pvalues_arr, exposed_prob)
            (
                self.impression_ids_f,
                self.slots_f,
                self.pv_costs_f,
                self.eff_costs_with_up,
                self.eff_pvs_with_up,
                self.costs,
                self.pvs,
                self.costs_next_slot,
            ) = self.prepare_impressions()
            self.time_steps_arr_f = time_arr[self.impression_ids_f]
            self.imp_idx_arr = impression_indices[self.impression_ids_f]
        else:
            valid_indices = self.time_steps_arr_f >= self.time_step
            self.impression_ids_f = self.impression_ids_f[valid_indices]
            self.slots_f = self.slots_f[valid_indices]
            self.pv_costs_f = self.pv_costs_f[valid_indices]
            self.eff_costs_with_up = self.eff_costs_with_up[valid_indices]
            self.eff_pvs_with_up = self.eff_pvs_with_up[valid_indices]
            self.costs = self.costs[valid_indices]
            self.costs_next_slot = self.costs_next_slot[valid_indices]
            self.pvs = self.pvs[valid_indices]
            self.time_steps_arr_f = self.time_steps_arr_f[valid_indices]
            self.imp_idx_arr = self.imp_idx_arr[valid_indices]

        cum_eff_cost = self.total_cost + np.cumsum(self.eff_costs_with_up)
        cum_eff_pv = self.total_conversions + np.cumsum(self.eff_pvs_with_up)
        cum_cpa = cum_eff_cost / (cum_eff_pv + self.EPS)
        cum_score = (
            cum_eff_pv
            * np.minimum(1, self.target_cpa / cum_cpa) ** 2
            * (cum_eff_cost <= self.total_budget)
        )
        max_score_idx = np.argmax(cum_score)
        max_score_mask = np.arange(len(self.impression_ids_f)) <= max_score_idx
        this_ts_mask = self.time_steps_arr_f == self.time_step
        valid_mask = np.logical_and(this_ts_mask, max_score_mask)

        good_imp_idx = self.imp_idx_arr[valid_mask]
        pvs_t = self.episode_pvalues_df[
            self.episode_pvalues_df.timeStepIndex == self.time_step
        ].pValue.item()
        num_imp = pvs_t.shape[0]
        good_cost_pv_max = safe_max(
            self.costs[max_score_mask] / self.pvs[max_score_mask]
        )  # to provide a ref value

        # Default: slightly below the most "inefficient" good impression
        action = np.ones(num_imp) * good_cost_pv_max / self.target_cpa - self.EPS

        # It works because even if some indices are repeated, we want to use the cost and the pv of
        # the least efficient slot among the selected ones,  and it always comes last in the good_imp_idx
        action[good_imp_idx] = (
            (
                self.cost_weight * self.costs[valid_mask]
                + (1 - self.cost_weight) * self.costs_next_slot[valid_mask]
            )
            / self.pvs[valid_mask]
            / self.target_cpa
        )
        if self.two_slopes_action:
            const_val = good_cost_pv_max / self.target_cpa
            y_0 = 1 / const_val
            y = 1 / action[good_imp_idx] - 1 / const_val
            x = pvs_t[good_imp_idx]
            if len(x) < 3:
                slope = 0
                intercept = 0
                x_0 = 0
            else:
                slope, intercept = np.polyfit(x, y, 1)
                x_0 = -intercept / slope
                log_y_0 = np.log(y_0)
            x_0 = np.clip(x_0 * 1e3, 0, 10)
            slope = np.clip(slope * 1e-3, 0, 10)
            oracle_action = np.array([log_y_0, x_0, slope])
        else:
            action = np.log(action)
            oracle_action = np.zeros((num_imp, len(self.act_keys)))
            oracle_action[:, self.pvalues_key_pos] = action
        return oracle_action

    def prepare_impressions(self):
        slot_3_eff_cost = self.eff_cost_table_f[:, 0]
        slot_2_eff_cost = self.eff_cost_table_f[:, 1]
        slot_1_eff_cost = self.eff_cost_table_f[:, 2]
        slot_3_cost = self.cost_table[:, 0]
        slot_2_cost = self.cost_table[:, 1]
        slot_1_cost = self.cost_table[:, 2]
        upgrade_3_2_cost = slot_2_eff_cost - slot_3_eff_cost
        upgrade_2_1_cost = slot_1_eff_cost - slot_2_eff_cost
        upgrade_3_1_cost = slot_1_eff_cost - slot_3_eff_cost
        slot_3_eff_pv = self.eff_pv_table_f[:, 0]
        slot_2_eff_pv = self.eff_pv_table_f[:, 1]
        slot_1_eff_pv = self.eff_pv_table_f[:, 2]
        slot_3_pv = self.pv_table[:, 0]
        slot_2_pv = self.pv_table[:, 1]
        slot_1_pv = self.pv_table[:, 2]
        upgrade_3_2_pv = slot_2_eff_pv - slot_3_eff_pv
        upgrade_2_1_pv = slot_1_eff_pv - slot_2_eff_pv
        upgrade_3_1_pv = slot_1_eff_pv - slot_3_eff_pv
        slot_3_ratio = slot_3_eff_pv / slot_3_eff_cost
        upgrade_3_2_ratio = upgrade_3_2_pv / upgrade_3_2_cost
        upgrade_2_1_ratio = upgrade_2_1_pv / upgrade_2_1_cost
        upgrade_3_1_ratio = upgrade_3_1_pv / upgrade_3_1_cost

        # # Sanity checks
        # # No 3_2 upgrade should be better than slot 3 (see theory)
        # assert (upgrade_3_2_ratio >= slot_3_ratio).mean() == 0
        # # No 3_1 upgrade should be better than slot 3 (see theory)
        # assert (upgrade_3_1_ratio >= slot_3_ratio).mean() == 0
        # # If 2_1 is better than 3_2, then 3_1 should be better than 3_2 (see theory)
        # assert ((upgrade_2_1_ratio > upgrade_3_2_ratio) == (upgrade_3_1_ratio > upgrade_3_2_ratio)).all()

        mask = upgrade_3_2_ratio > upgrade_2_1_ratio

        flat_ratio = np.concatenate(
            (
                slot_3_ratio,
                upgrade_3_2_ratio[mask],
                upgrade_2_1_ratio[mask],
                upgrade_3_1_ratio[~mask],
            )
        )
        flat_cost = np.concatenate(
            (slot_3_cost, slot_2_cost[mask], slot_1_cost[mask], slot_1_cost[~mask])
        )
        flat_cost_next_slot = np.concatenate(
            (
                slot_2_cost,
                slot_1_cost[mask],
                slot_1_cost[mask] * 1.2,
                slot_1_cost[~mask] * 1.2,
            )
        )
        flat_eff_cost_with_up = np.concatenate(
            (
                slot_3_eff_cost,
                upgrade_3_2_cost[mask],
                upgrade_2_1_cost[mask],
                upgrade_3_1_cost[~mask],
            )
        )
        flat_pv = np.concatenate(
            (slot_3_pv, slot_2_pv[mask], slot_1_pv[mask], slot_1_pv[~mask])
        )
        flat_eff_pv_with_up = np.concatenate(
            (
                slot_3_eff_pv,
                upgrade_3_2_pv[mask],
                upgrade_2_1_pv[mask],
                upgrade_3_1_pv[~mask],
            )
        )
        slot_indices = np.concatenate(
            (
                np.zeros_like(slot_3_ratio),
                np.ones_like(upgrade_3_2_ratio[mask]),
                2 * np.ones_like(upgrade_2_1_ratio[mask]),
                2 * np.ones_like(upgrade_3_1_ratio[~mask]),
            )
        ).astype(int)
        all_imp = np.arange(self.eff_cost_table_f.shape[0])
        impression_indices = np.concatenate(
            (all_imp, all_imp[mask], all_imp[mask], all_imp[~mask])
        )

        # Exclude pv equal to 0 - never good and could cause issues in ranking
        valid_mask = flat_eff_pv_with_up > 0
        flat_ratio = flat_ratio[valid_mask]
        flat_cost = flat_cost[valid_mask]
        flat_cost_next_slot = flat_cost_next_slot[valid_mask]
        flat_eff_cost_with_up = flat_eff_cost_with_up[valid_mask]
        flat_pv = flat_pv[valid_mask]
        flat_eff_pv_with_up = flat_eff_pv_with_up[valid_mask]
        slot_indices = slot_indices[valid_mask]
        impression_indices = impression_indices[valid_mask]

        # Rank all impressions by pv/cost ratio (descending)
        sorted_indices = np.argsort(-flat_ratio)  # negative sign for descending order

        # Return the sorted indices along with corresponding impression and slot
        sorted_impression_indices = impression_indices[sorted_indices]
        sorted_slot_indices = slot_indices[sorted_indices]
        sorted_flat_ratio = flat_ratio[sorted_indices]
        sorted_flat_cost = flat_cost[sorted_indices]
        sorted_flat_cost_next_slot = flat_cost_next_slot[sorted_indices]
        sorted_flat_eff_cost_with_up = flat_eff_cost_with_up[sorted_indices]
        sorted_flat_pv = flat_pv[sorted_indices]
        sorted_flat_eff_pv_with_up = flat_eff_pv_with_up[sorted_indices]

        return (
            sorted_impression_indices,
            sorted_slot_indices,
            sorted_flat_ratio,
            sorted_flat_eff_cost_with_up,
            sorted_flat_eff_pv_with_up,
            sorted_flat_cost,
            sorted_flat_pv,
            sorted_flat_cost_next_slot,
        )

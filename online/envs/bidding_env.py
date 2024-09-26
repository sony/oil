import gymnasium as gym
import numpy as np
import pandas as pd
from .helpers import safe_mean


class BiddingEnv(gym.Env):
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

    DEFAULT_ACT_KEYS = [
        "pvalue",
    ]

    DEFAULT_RWD_WEIGHTS = {
        "dense": 0.0,
        "sparse": 1.0,
    }
    EPS = 1e-6

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
        pvalues_df_path=None,
        bids_df_path=None,
        budget_range=(6000, 6000),
        target_cpa_range=(8, 8),
        advertiser_id=None,
        obs_keys=DEFAULT_OBS_KEYS,
        rwd_weights=DEFAULT_RWD_WEIGHTS,
        act_keys=DEFAULT_ACT_KEYS,
        new_action=False,
        multi_action=False,  # Deprecated, select the action keys instead
        exp_action=False,
        sample_log_budget=False,
        simplified_bidding=False,
        auction_noise=0,
        pvalues_rescale_range=(1, 1),
        simplified_exposure_prob_range=(1, 1),
        stochastic_exposure=False,
        deterministic_conversion=False,
        seed=0,
    ):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(obs_keys),), dtype=np.float32
        )
        if multi_action:
            print(
                "Warning: multi_action is deprecated, use act_keys instead. Ignoring act_keys."
            )
            self.act_keys = [
                "pvalue",
                "pvalue_sigma",
                "pvalue_square",
                "pvalue_sigma_square",
                "pvalue_sigma_pvalue",
                "pvalue_sqrt",
            ]
        else:
            self.act_keys = act_keys
        self.pvalues_key_pos = self.act_keys.index("pvalue")
        if new_action:
            self.action_space = gym.spaces.Box(
                low=-10, high=10, shape=(len(self.act_keys),), dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=0, high=10, shape=(len(self.act_keys),), dtype=np.float32
            )
        if exp_action:
            assert new_action, "Exponential action requires new action"
        self.obs_keys = obs_keys
        self.rwd_weights = rwd_weights
        self.new_action = new_action
        self.exp_action = exp_action
        self.sample_log_budget = sample_log_budget
        self.simplified_bidding = simplified_bidding
        self.auction_noise = auction_noise
        self.pvalues_rescale_min, self.pvalues_rescale_max = pvalues_rescale_range
        self.simplified_exposure_prob_min, self.simplified_exposure_prob_max = (
            simplified_exposure_prob_range
        )
        self.stochastic_exposure = stochastic_exposure
        self.deterministic_conversion = deterministic_conversion

        if pvalues_df_path is None or bids_df_path is None:
            print("Warning: creating a dummy environment with no dataset")
        else:
            self.pvalues_df = self.load_pvalues_df(pvalues_df_path)
            self.bids_df = self.load_bids_df(bids_df_path)
            self.episode_length = len(self.bids_df.timeStepIndex.unique())
            if advertiser_id is None:
                self.advertiser_list = list(self.pvalues_df.advertiserNumber.unique())
            else:
                self.advertiser_list = [advertiser_id]
            categories_df = self.pvalues_df.groupby(
                "advertiserNumber"
            ).advertiserCategoryIndex.first()
            self.advertiser_category_dict = {
                advertiser: category
                for advertiser, category in zip(
                    categories_df.index, categories_df.values
                )
            }
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
        self.history_info = {key: [] for key in self.HISTORY_KEYS}
        self.history_info.update({key: [] for key in self.NO_HISTORY_KEYS})
        self.history_info.update({key: [] for key in self.HISTORY_AND_SLOT_KEYS})
        self.history_info.update(
            {
                f"{key}_slot_{slot}": []
                for key in self.HISTORY_AND_SLOT_KEYS
                for slot in range(1, 4)
            }
        )

        self.pvalues_rescale_coef = self.np_random.uniform(
            self.pvalues_rescale_min, self.pvalues_rescale_max
        )
        self.simplified_exposure_prob = self.np_random.uniform(
            self.simplified_exposure_prob_min, self.simplified_exposure_prob_max
        )
        self.episode_bids_df = self.get_episode_bids_df()
        self.episode_pvalues_df = self.get_episode_pvalues_df()
        self.ranked_df = None
        self.impression_ids = None
        self.slots = None
        self.pv_costs = None
        self.time_steps_arr = None
        self.eff_pv_table = None
        self.eff_cost_table = None

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
        state_dict = self.get_state_dict(pvalues)
        state = self.get_state(state_dict)
        return state, {}

    def step(self, action):
        bid_data = self.get_bid_data()

        # Get current pvalues to compute the bids
        pvalues, pvalues_sigma = self.get_pvalues_mean_and_std()

        bid_coef, alpha = self.compute_bid_coef(action, pvalues, pvalues_sigma)
        advertiser_bids = bid_coef * self.target_cpa
        top_bids = bid_data.bid.item()

        top_bids_exposed = bid_data.isExposed.item()
        if self.stochastic_exposure:
            # We simulate exposure as a Bernoulli with the mean exposure probability per slot
            # It could be unrealistic that one bid is not exposed but a lower one is, should be fine for training
            exposure_prob_per_slot = np.mean(top_bids_exposed, axis=0)
            top_bids_exposed = self.np_random.binomial(
                n=1, p=exposure_prob_per_slot, size=top_bids_exposed.shape
            )

        top_bids_cost = bid_data.cost.item()
        least_winning_cost = top_bids_cost[:, 0]

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
            history_info_update[key] = slot_info["func"](slot_info["data"])
            for slot in range(3):
                func = slot_info["func"]
                condition = slot_info["condition"]
                history_info_update[f"{key}_slot_{3 - slot}"] = func(
                    slot_info["data"][np.logical_and(bid_position == slot, condition)]
                )

        for key, value in history_info_update.items():
            self.history_info[key].append(value)

        info = {
            "action": alpha,
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
        state_dict = self.get_state_dict(new_pvalues)
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

        # TODO: check if the conversion depends on the position (AC: small dependence, we ignore it for now)
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
        if self.simplified_bidding:
            bid_success = advertiser_bids >= least_winning_cost
            bid_cost = least_winning_cost * bid_success
            bid_position = np.zeros_like(
                bid_cost
            )  # No bid position in simplified bidding
            # bid_exposed = (
            #     bid_success  # Simplified bidding always exposes successful bids
            # )
            bid_exposed = (
                self.np_random.binomial(
                    n=1, p=self.simplified_exposure_prob, size=bid_success.shape
                )
                * bid_success
            )

        else:
            advertiser_bid_higher = advertiser_bids[:, None] >= top_bids
            bid_success = advertiser_bid_higher.any(axis=1)
            bid_position = np.sum(advertiser_bid_higher, axis=1) - 1

            # Exposed is 0 if the bid is not successful
            bid_exposed = np.zeros_like(bid_position)
            bid_exposed[bid_success] = top_bids_exposed[
                bid_success, bid_position[bid_success]
            ]

            # If I am higher than a bid, I pay that bid's price. No payment for not winning or not exposing
            bid_cost = (
                top_bids[np.arange(len(bid_position)), bid_position] * bid_exposed
            )
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
        if self.sample_log_budget:
            l = np.log(self.target_cpa_range[0])
            h = np.log(self.target_cpa_range[1])
            return np.exp(self.np_random.uniform(l, h))
        else:
            return self.np_random.uniform(*self.target_cpa_range)

    def sample_advertiser(self):
        return self.np_random.choice(self.advertiser_list)

    def sample_period(self):
        return self.np_random.choice(self.period_list)

    def get_state_dict(self, pvalues):
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
        state = np.array([state_dict[key] for key in self.obs_keys]).astype(np.float32)
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
        action = np.atleast_1d(action).copy()
        if self.new_action:
            if self.exp_action:
                action[0] = np.exp(action[0])
            else:
                action[0] = action[0] + 1

        bid_basis = self.get_bid_basis(pvalues, pvalues_sigma)
        bid_coef = np.clip(np.dot(action, bid_basis.T), 0, np.inf)
        alpha = np.sum(bid_coef) / (np.sum(pvalues) + self.EPS)
        return bid_coef, alpha

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
        print(f"Loading bids from {bids_df_path}")
        bids_df = pd.read_parquet(bids_df_path)
        bids_df["bid"] = bids_df["bid"].apply(np.stack)
        bids_df["isExposed"] = bids_df["isExposed"].apply(np.stack)
        bids_df["cost"] = bids_df["cost"].apply(np.stack)
        return bids_df

    def get_baseline_action(self):
        remaining_budget_excess = (
            self.remaining_budget
            * self.episode_length
            / (self.total_budget * (self.episode_length - self.time_step + 1))
        )
        return 0.8 * remaining_budget_excess

    def noisy_bid_and_cost(self, row, noise):
        bid = row["bid"]
        cost = row["cost"]
        second_price_ratio = cost[:, 0] / bid[:, 0]

        # Add noise to bids
        noisy_bid = bid * (1 + self.np_random.uniform(-noise, noise, bid.shape))
        noisy_bid = np.sort(noisy_bid, axis=1)
        noisy_cost = np.zeros_like(noisy_bid)
        noisy_cost[:, 0] = noisy_bid[:, 0] * second_price_ratio
        noisy_cost[:, 1:] = noisy_bid[:, :-1]

        # Return the modified noisy_bid and noisy_cost
        return pd.Series([noisy_bid, noisy_cost])

    def get_episode_bids_df(self):
        ep_bids_df = self.bids_df[
            self.bids_df.deliveryPeriodIndex == self.period
        ].copy()
        if self.auction_noise > 0:
            ep_bids_df[["bid", "cost"]] = ep_bids_df.apply(
                lambda x: self.noisy_bid_and_cost(x, self.auction_noise), axis=1
            )
        ep_bids_df["bid"] = ep_bids_df["bid"] * self.pvalues_rescale_coef
        ep_bids_df["cost"] = ep_bids_df["cost"] * self.pvalues_rescale_coef
        return ep_bids_df

    def get_episode_pvalues_df(self):
        ep_pv_df = self.pvalues_df[
            (self.pvalues_df.advertiserNumber == self.advertiser)
            & (self.pvalues_df.deliveryPeriodIndex == self.period)
        ].copy()
        ep_pv_df["pValue"] = ep_pv_df["pValue"] * self.pvalues_rescale_coef
        return ep_pv_df

    def compute_ranked_impressions_df(self):
        # Extract pvalues and costs as lists of arrays
        pvalues_list = np.concatenate(self.episode_pvalues_df.pValue.values)
        min_cost_list = np.concatenate(
            self.episode_bids_df.cost.apply(lambda x: x[:, 0]).values
        )

        # Create the DataFrame directly from arrays without looping
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
        if self.simplified_bidding:
            return self.get_simplified_oracle_action()
        else:
            return self.get_realistic_oracle_action()

    def get_simplified_oracle_action(self):
        if self.ranked_df is None:
            self.ranked_df = self.compute_ranked_impressions_df()
        df_sorted = self.ranked_df[
            self.ranked_df.time_step >= self.time_step
        ].reset_index(drop=True)
        df_sorted["cum_conversions"] = (
            df_sorted["pvalue"].cumsum() * self.simplified_exposure_prob
            + self.total_conversions
        )  # If not exposed, no conversion
        df_sorted["cum_cost"] = (
            df_sorted["cost"].cumsum() * self.simplified_exposure_prob + self.total_cost
        )  # If not exposed, no cost
        df_sorted["cum_cpa"] = df_sorted["cum_cost"] / df_sorted["cum_conversions"]
        df_sorted["score"] = (
            df_sorted["cum_conversions"]
            * np.minimum(1, self.target_cpa / df_sorted["cum_cpa"]) ** 2
        )
        # We use total budget because the cost already includes the current total cost
        df_within_budget = df_sorted[df_sorted["cum_cost"] <= self.total_budget]

        if df_within_budget.empty:
            # We have run out of budget, it does not matter what we bid
            action = 1
        else:
            # Find the impressions that lead to the max score
            max_score_row = df_within_budget["score"].idxmax()
            selected_rows = df_sorted.loc[:max_score_row]

            # Select the action that buys the best impression opportunities
            action = 1 / selected_rows.pv_over_cost.min() / self.target_cpa
        if self.new_action:
            if self.exp_action:
                action = np.log(action)
            else:
                action = action - 1
        oracle_action = np.zeros(len(self.act_keys))
        oracle_action[self.pvalues_key_pos] = action
        return oracle_action

    def get_realistic_oracle_action(self):
        if self.impression_ids is None:
            # Sort the impression opportunities for all slots
            cost_table = np.vstack(self.episode_bids_df.bid)
            exposed_table = np.vstack(self.episode_bids_df.isExposed)
            pvalues_arr = np.concatenate(self.episode_pvalues_df.pValue.to_list())
            time_arr = np.repeat(
                np.arange(len(self.episode_pvalues_df)),
                self.episode_pvalues_df.pValue.apply(len),
            )
            exposed_prob = np.mean(exposed_table, axis=0)
            self.eff_cost_table = cost_table * exposed_prob
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
                action = 1
            else:
                action = 1 / best_pv_cost / self.target_cpa

        if self.new_action:
            if self.exp_action:
                action = np.log(action)
            else:
                action = action - 1
        oracle_action = np.zeros(len(self.act_keys))
        oracle_action[self.pvalues_key_pos] = action
        return oracle_action

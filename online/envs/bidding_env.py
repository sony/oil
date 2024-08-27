import gymnasium as gym
import numpy as np
import pandas as pd


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
        "last_threee_least_winning_cost_mean",
        "last_three_pvalues_mean",
        "last_three_conversion_mean",
        "last_three_bid_success_mean",
        "current_pvalues_mean",
        "current_pv_num",
        "last_three_pv_num",
        "pv_num_total",
    ]

    DEFAULT_RWD_WEIGHTS = {
        "dense": 0.0,
        "sparse": 1.0,
    }

    def __init__(
        self,
        pvalues_df_path,
        bids_df_path,
        budget_range,
        target_cpa_range,
        obs_keys=DEFAULT_OBS_KEYS,
        rwd_weights=DEFAULT_RWD_WEIGHTS,
        new_action=False,
        seed=0,
    ):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(obs_keys),), dtype=np.float32
        )
        if new_action:
            self.action_space = gym.spaces.Box(
                low=-1, high=10, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=0, high=10, shape=(1,), dtype=np.float32
            )
        self.new_action = new_action
        self.pvalues_df = self.load_pvalues_df(pvalues_df_path)
        self.bids_df = self.load_bids_df(bids_df_path)
        self.episode_length = len(self.bids_df.timeStepIndex.unique())
        self.advertiser_list = list(self.pvalues_df.advertiserNumber.unique())
        self.period_list = list(self.pvalues_df.deliveryPeriodIndex.unique())
        self.budget_range = budget_range
        self.target_cpa_range = target_cpa_range
        self.obs_keys = obs_keys
        self.rwd_weights = rwd_weights
        self.reset(seed=seed)

    def reset_campaign_params(self, budget=None, target_cpa=None):
        self.advertiser = self.sample_advertiser()
        self.period = self.sample_period()
        self.total_budget = self.sample_budget() if budget is None else budget
        self.target_cpa = self.sample_cpa() if target_cpa is None else target_cpa
        self.time_step = 0
        self.remaining_budget = self.total_budget
        self.mean_bid_list = []
        self.mean_pvalues_list = []
        self.mean_least_winning_cost_list = []
        self.mean_conversion_list = []
        self.mean_bid_success_list = []
        self.num_pv_list = []
        self.total_conversions = 0
        self.total_cost = 0

    def reset(self, budget=None, target_cpa=None, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_campaign_params(budget, target_cpa)
        pvalues, pvalues_std = self.get_pvalues_mean_and_std()
        state = self.get_state(pvalues)

        return state, {}

    def step(self, action):
        bid_data = self.get_bid_data()
        pvalues, pvalues_sigma = self.get_pvalues_mean_and_std()
        if self.new_action:
            action = action + 1
        advertiser_bids = action * pvalues * self.target_cpa
        top_bids = bid_data.bid.item()

        # TODO: we should simulate exposure as a Beroulli
        top_bids_exposed = bid_data.isExposed.item()

        bid_success, bid_cost, bid_conversion = self.simulate_ad_bidding(
            pvalues, pvalues_sigma, advertiser_bids, top_bids, top_bids_exposed
        )

        self.mean_bid_list.append(np.mean(advertiser_bids))
        self.mean_pvalues_list.append(np.mean(pvalues))
        self.mean_least_winning_cost_list.append(
            np.mean(top_bids[0])
        )  # only the smallest bid
        self.mean_conversion_list.append(np.mean(bid_conversion))

        self.mean_bid_success_list.append(np.mean(bid_success))
        self.num_pv_list.append(len(pvalues))
        self.time_step += 1
        self.total_conversions += np.sum(bid_conversion)
        self.total_cost += np.sum(bid_cost)
        self.remaining_budget -= np.sum(bid_cost)
        terminated = self.time_step >= self.episode_length
        dense_reward = self.compute_score(np.sum(bid_cost), np.sum(bid_conversion))

        info = {
            "action": action,
            "bid": np.mean(advertiser_bids),
        }
        if terminated:
            cpa = (
                self.total_cost / self.total_conversions
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
                "avg_pvalues": np.mean(self.mean_pvalues_list),
                "score_over_pvalue": sparse_reward / np.mean(self.mean_pvalues_list),
                "score_over_budget": sparse_reward / self.total_budget,
                "score_over_cpa": sparse_reward / cpa if cpa > 0 else 0,
                "cost_over_budget": self.total_cost / self.total_budget,
                "target_cpa_over_cpa": self.target_cpa / cpa if cpa > 0 else 0,
                "score": sparse_reward,
            }
            info.update(episode_end_info)
            info.update(reward_dict)
            reward = self.compute_reward(reward_dict)
        else:
            reward_dict = {"sparse": 0, "dense": dense_reward}
            info.update(reward_dict)
            reward = self.compute_reward(reward_dict)
        state = self.get_state(pvalues)
        return state, reward, terminated, False, info

    def compute_score(self, cost, conversions):
        cpa = cost / conversions if conversions > 0 else 0
        cpa_coeff = min(1, (self.target_cpa / cpa) ** 2) if cpa > 0 else 0
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
    ):
        bid_success, bid_exposed, bid_cost = self.compute_success_exposition_cost(
            advertiser_bids, top_bids, top_bids_exposed
        )

        bid_success, bid_exposed, bid_cost = self.handle_overcost(
            bid_success,
            bid_exposed,
            bid_cost,
            advertiser_bids,
            top_bids,
            top_bids_exposed,
        )

        # TODO: check if the conversion depends on the position (AC: small dependence, we ignore it for now)
        pvalues_sampled = np.clip(self.np_random.normal(pvalues, pvalues_sigma), 0, 1)
        bid_conversion = self.np_random.binomial(n=1, p=pvalues_sampled) * bid_exposed
        return bid_success, bid_cost, bid_conversion

    def compute_success_exposition_cost(
        self, advertiser_bids, top_bids, top_bids_exposed
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
        return bid_success, bid_exposed, bid_cost

    def handle_overcost(
        self,
        bid_success,
        bid_exposed,
        bid_cost,
        advertiser_bids,
        top_bids,
        top_bids_exposed,
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
            bid_success, bid_exposed, bid_cost = self.compute_success_exposition_cost(
                advertiser_bids, top_bids, top_bids_exposed
            )
            total_cost = sum(bid_cost)
        return bid_success, bid_exposed, bid_cost

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

    def get_state_dict(self, pvalues):
        if self.time_step == 0:
            return {
                "time_left": 1,
                "budget_left": 1,
                "historical_bid_mean": 0,
                "last_three_bid_mean": 0,
                "least_winning_cost_mean": 0,
                "pvalues_mean": 0,
                "conversion_mean": 0,
                "bid_success_mean": 0,
                "last_threee_least_winning_cost_mean": 0,
                "last_three_pvalues_mean": 0,
                "last_three_conversion_mean": 0,
                "last_three_bid_success_mean": 0,
                "current_pvalues_mean": 0,
                "current_pv_num": 0,
                "last_three_pv_num": 0,
                "pv_num_total": 0,
            }
        else:
            state_dict = {
                "time_left": (self.episode_length - self.time_step)
                / self.episode_length,
                "budget_left": max(self.remaining_budget, 0) / self.total_budget,
                "historical_bid_mean": np.mean(self.mean_bid_list),
                "last_three_bid_mean": np.mean(self.mean_bid_list[-3:]),
                "least_winning_cost_mean": np.mean(self.mean_least_winning_cost_list),
                "pvalues_mean": np.mean(self.mean_pvalues_list),
                "conversion_mean": np.mean(self.mean_conversion_list),
                "bid_success_mean": np.mean(self.mean_bid_success_list),
                "last_threee_least_winning_cost_mean": np.mean(
                    self.mean_least_winning_cost_list[-3:]
                ),
                "last_three_pvalues_mean": np.mean(self.mean_pvalues_list[-3:]),
                "last_three_conversion_mean": np.mean(self.mean_conversion_list[-3:]),
                "last_three_bid_success_mean": np.mean(self.mean_bid_success_list[-3:]),
                "current_pvalues_mean": np.mean(pvalues),
                "current_pv_num": len(pvalues),
                "last_three_pv_num": sum(self.num_pv_list[-3:]),
                "pv_num_total": sum(self.num_pv_list),
            }
        return state_dict

    def get_state(self, pvalues):
        state_dict = self.get_state_dict(pvalues)
        state = np.array([state_dict[key] for key in self.obs_keys]).astype(np.float32)
        return state

    def get_pvalues_mean_and_std(self):
        p_row = self.pvalues_df[
            (self.pvalues_df.advertiserNumber == self.advertiser)
            & (self.pvalues_df.deliveryPeriodIndex == self.period)
            & (self.pvalues_df.timeStepIndex == self.time_step)
        ]
        return p_row.pValue.item(), p_row.pValueSigma.item()

    def get_bid_data(self):
        bid_data = self.bids_df[
            (self.bids_df.deliveryPeriodIndex == self.period)
            & (self.bids_df.timeStepIndex == self.time_step)
        ]
        return bid_data

    def load_pvalues_df(self, pvalues_df_path):
        print(f"Loading pvalues from {pvalues_df_path}")
        return pd.read_parquet(pvalues_df_path)

    def load_bids_df(self, bids_df_path):
        print(f"Loading bids from {bids_df_path}")
        bids_df = pd.read_parquet(bids_df_path)
        bids_df["bid"] = bids_df["bid"].apply(np.stack)
        bids_df["isExposed"] = bids_df["isExposed"].apply(np.stack)
        return bids_df

    def get_baseline_action(self):
        remaining_budget_excess = (
            self.remaining_budget
            * self.episode_length
            / (self.total_budget * (self.episode_length - self.time_step))
        )
        return remaining_budget_excess
        

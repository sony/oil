import numpy as np
import math
import logging
import pathlib
import shutil
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
from bidding_train_env.strategy.bidding_strategy_factory import BiddingStrategyFactory
from definitions import ROOT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test(
    data_path=ROOT_DIR / "data/traffic/period-13.csv",
    budget=3000,
    target_cpa=8,
    category=0,
    experiment_path=ROOT_DIR / "saved_model" / "BC" / "train_004" / "checkpoint_10000",
    strategy_name="bc",
    device="cpu",
):
    """
    offline evaluation
    """
    if not experiment_path.exists():
        # Check if there is the same file in the output directory instead of the saved_model directory
        output_path = ROOT_DIR / "output" / experiment_path.relative_to(ROOT_DIR / "saved_model")
        
        # Copy the file from output_path to experiment_path
        if output_path.exists():
            shutil.copytree(output_path, experiment_path)
        else:
            raise FileNotFoundError(
                f"Model file not found in {experiment_path} or {output_path}"
            )

    data_loader = TestDataLoader(file_path=data_path)
    env = OfflineEnv()
    agent = BiddingStrategyFactory.create(
        strategy_name=strategy_name,
        budget=budget,
        cpa=target_cpa,
        category=category,
        experiment_path=experiment_path,
        device=device,
    )
    print(agent.name)

    keys, test_dict = data_loader.keys, data_loader.test_dict
    key = keys[0]
    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(
        key
    )
    rewards = np.zeros(num_timeStepIndex)
    history = {
        "historyBids": [],
        "historyAuctionResult": [],
        "historyImpressionResult": [],
        "historyLeastWinningCost": [],
        "historyPValueInfo": [],
    }

    for timeStep_index in range(num_timeStepIndex):
        # logger.info(f"Timestep Index: {timeStep_index + 1} Begin")

        pValue = pValues[timeStep_index]
        pValueSigma = pValueSigmas[timeStep_index]
        leastWinningCost = leastWinningCosts[timeStep_index]

        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pValue.shape[0])
        else:

            bid = agent.bidding(
                timeStep_index,
                pValue,
                pValueSigma,
                history["historyPValueInfo"],
                history["historyBids"],
                history["historyAuctionResult"],
                history["historyImpressionResult"],
                history["historyLeastWinningCost"],
            )

        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
            pValue, pValueSigma, bid, leastWinningCost
        )

        # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
        over_cost_ratio = max(
            (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0
        )
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(
                pv_index,
                int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                replace=False,
            )
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = (
                env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost)
            )
            over_cost_ratio = max(
                (np.sum(tick_cost) - agent.remaining_budget)
                / (np.sum(tick_cost) + 1e-4),
                0,
            )

        agent.remaining_budget -= np.sum(tick_cost)
        rewards[timeStep_index] = np.sum(tick_conversion)
        temHistoryPValueInfo = [
            (pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])
        ]
        history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(leastWinningCost)
        temAuctionResult = np.array(
            [
                (tick_status[i], tick_status[i], tick_cost[i])
                for i in range(tick_status.shape[0])
            ]
        )
        history["historyAuctionResult"].append(temAuctionResult)
        temImpressionResult = np.array(
            [(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])]
        )
        history["historyImpressionResult"].append(temImpressionResult)
        # logger.info(f"Timestep Index: {timeStep_index + 1} End")
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    score = getScore_nips(all_reward, cpa_real, cpa_constraint)

    logger.info(f"Total Reward: {all_reward}")
    logger.info(f"Total Cost: {all_cost}")
    logger.info(f"CPA-real: {cpa_real}")
    logger.info(f"CPA-constraint: {cpa_constraint}")
    logger.info(f"Score: {score}")


if __name__ == "__main__":
    run_test()

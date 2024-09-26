import numpy as np
import math
import logging
import json
import shutil
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
from bidding_train_env.strategy.bidding_strategy_factory import BiddingStrategyFactory
from definitions import ROOT_DIR
from itertools import product

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
    data_path_or_dataloader=ROOT_DIR / "data/raw_traffic_final_parquet/period-27.parquet",
    budget_list=[2000, 3000, 4000, 5000],
    target_cpa_list=[60, 95, 130],
    category_list=[0],
    experiment_path="ONBC/020_onbc_seed_0_transformer_new_data_realistic",  # "BC/train_whole_dataset_009/checkpoint_8000",
    checkpoint=4500000,
    strategy_name="onbc_transformer",  # ppo, onbc_transformer, onlineLp
    device="cpu",
    algo="onbc_transformer",  # onbc, onbc_transformer
):
    """
    offline evaluation
    """
    full_experiment_path = ROOT_DIR / "saved_model" / experiment_path
    if not full_experiment_path.exists():
        # Check if there is the same file in the output directory instead of the saved_model directory
        output_path = ROOT_DIR / "output" / experiment_path

        # Copy the file from output_path to experiment_path
        if output_path.exists():
            shutil.copytree(output_path, experiment_path)
        else:
            raise FileNotFoundError(
                f"Model file not found in {experiment_path} or {output_path}"
            )

    if isinstance(data_path_or_dataloader, TestDataLoader):
        data_loader = data_path_or_dataloader
    else:
        logger.info(f"Loading data from {data_path_or_dataloader}")
        data_loader = TestDataLoader(file_path=data_path_or_dataloader)
        logger.info(f"Data loaded successfully")

    env = OfflineEnv()

    result_dict_list = []
    # Test on all combinations of budget, target_cpa, and category
    for budget, target_cpa, category in product(
        budget_list, target_cpa_list, category_list
    ):
        agent = BiddingStrategyFactory.create(
            strategy_name=strategy_name,
            budget=budget,
            cpa=target_cpa,
            category=category,
            experiment_path=full_experiment_path,
            checkpoint=checkpoint,
            device=device,
            algo=algo,
        )

        keys = data_loader.keys
        key = keys[0]
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = (
            data_loader.mock_data(key)
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

            tick_value, tick_cost, tick_status, tick_conversion = (
                env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost)
            )

            # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
            over_cost_ratio = max(
                (np.sum(tick_cost) - agent.remaining_budget)
                / (np.sum(tick_cost) + 1e-4),
                0,
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
                [
                    (tick_conversion[i], tick_conversion[i])
                    for i in range(pValue.shape[0])
                ]
            )
            history["historyImpressionResult"].append(temImpressionResult)
            # logger.info(f"Timestep Index: {timeStep_index + 1} End")
            all_reward = np.sum(rewards)
            all_cost = agent.budget - agent.remaining_budget
            cpa_real = all_cost / (all_reward + 1e-10)
            cpa_constraint = agent.cpa
            score = getScore_nips(all_reward, cpa_real, cpa_constraint)

        logger.info(f"Category: {category}")
        logger.info(f"CPA-constraint: {cpa_constraint}")
        logger.info(f"Budget: {budget}")
        logger.info(f"Total Reward: {all_reward}")
        logger.info(f"Total Cost: {all_cost}")
        logger.info(f"CPA-real: {cpa_real}")
        logger.info(f"Score: {score}")
        result_dict_list.append(
            {
                "category": category,
                "cpa_constraint": cpa_constraint,
                "budget": budget,
                "total_reward": all_reward,
                "total_cost": all_cost,
                "cpa_real": cpa_real,
                "score": score,
            }
        )
    cost_budget_ratio = np.mean(
        [d["total_cost"] / d["budget"] for d in result_dict_list]
    )
    target_real_cpa_ratio = np.mean(
        [d["cpa_constraint"] / d["cpa_real"] for d in result_dict_list]
    )
    score = np.mean([d["score"] for d in result_dict_list])
    reward = np.mean([d["total_reward"] for d in result_dict_list])

    logger.info("Overall average results:")
    logger.info(f"Cost/Budget Ratio: {cost_budget_ratio}")
    logger.info(f"Target/Real CPA Ratio: {target_real_cpa_ratio}")
    logger.info(f"Score: {score}")
    logger.info(f"Reward: {reward}")

    # Turn the path after saved_model into a string and use it as the experiment name
    out_path = (
        ROOT_DIR
        / "output"
        / "offline_evaluation"
        / experiment_path
        / f"checkpoint_{checkpoint}.json"
    )
    logger.info(f"Saving results to {out_path}")
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result_dict_list, f, indent=4)


if __name__ == "__main__":
    run_test()

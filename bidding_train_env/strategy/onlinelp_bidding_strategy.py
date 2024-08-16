import pandas as pd
import os
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from definitions import ROOT_DIR


class OnlineLpBiddingStrategy(BaseBiddingStrategy):
    """
    OnlineLpBidding Strategy
    """

    def __init__(
        self,
        budget=100,
        name="OnlineLpBiddingStrategy",
        cpa=2,
        category=1,
        experiment_path=ROOT_DIR / "saved_model" / "onlineLpTest",
        device="cpu",
    ):
        super().__init__(budget, name, cpa, category)
        model_path = experiment_path / "period.csv"
        self.category = category

        self.model = pd.read_csv(model_path)

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
        tem = self.model[
            (self.model["timeStepIndex"] == timeStepIndex)
            & (self.model["advertiserCategoryIndex"] == self.category)
        ]
        alpha = self.cpa
        if len(tem) == 0:
            pass
        else:

            def find_first_cpa_above_budget(df, budget):
                filtered_df = df[df["cum_cost"] > budget]

                if not filtered_df.empty:
                    return filtered_df.iloc[0]["realCPA"]
                else:
                    return None

            res = find_first_cpa_above_budget(tem, self.remaining_budget)
            if res is None:
                pass
            else:
                alpha = res

        alpha = min(self.cpa * 1.5, alpha)
        bids = alpha * pValues

        # bids = 5 * self.cpa * pValues if self.remaining_budget >= 0 else 0

        remaining_budget_excess = (
            self.remaining_budget * 48 / (self.budget * (48 - timeStepIndex))
        )
        # used_budget_defect =
        # print(remaining_budget_excess)
        # print(used_budget_defect)
        # bids = 0.8 * self.cpa * remaining_budget_excess * used_budget_defect * pValues if self.remaining_budget >= 0 else 0
        bids = (
            1 * self.cpa * remaining_budget_excess * pValues
            if self.remaining_budget >= 0
            else 0
        )
        return bids

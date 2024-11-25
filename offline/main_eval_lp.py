def find_closest_budget(budget, available_budgets):
    """
    Find the closest budget from the available pre-computed budgets.
    """
    return min(available_budgets, key=lambda x: abs(x - budget))

def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
            historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
    """
    Bids for all the opportunities in a delivery period, using the closest precomputed budget.
    """
    # Available budgets from the precomputed table
    available_budgets = self.model['budget'].unique()
    closest_budget = find_closest_budget(self.remaining_budget, available_budgets)
    
    tem = self.model[
        (self.model["timeStepIndex"] == timeStepIndex) &
        (self.model["advertiserCategoryIndex"] == self.category) &
        (self.model["budget"] == closest_budget)
    ]
    
    alpha = self.cpa
    if len(tem) == 0:
        pass
    else:
        def find_first_cpa_above_budget(df, budget):
            filtered_df = df[df['cum_cost'] > budget]
            if not filtered_df.empty:
                return filtered_df.iloc[0]['realCPA']
            else:
                return None

        res = find_first_cpa_above_budget(tem, self.remaining_budget)
        if res is not None:
            alpha = res

    alpha = min(self.cpa * 1.5, alpha)
    bids = alpha * pValues
    return bids

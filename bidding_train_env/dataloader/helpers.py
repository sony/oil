import numpy as np
from online.envs.helpers import safe_mean

def get_score_neurips(total_conversions, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * total_conversions


def compute_alpha(ad_df, ts, target_cpa):
    bids = ad_df[ad_df.timeStepIndex == ts].bid.to_numpy()
    pvalues = ad_df[ad_df.timeStepIndex == ts].pValue.to_numpy()
    alpha = safe_mean(bids[pvalues > 0] / pvalues[pvalues > 0]) / target_cpa
    return np.log(alpha) if alpha > np.exp(-10) else -10

from gymnasium.envs.registration import register


register(
        id="BiddingEnv-v0",
        entry_point="envs.bidding_env:BiddingEnv",
    )
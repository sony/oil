from bidding_train_env.strategy.player_bidding_strategy import PlayerBiddingStrategy
from bidding_train_env.strategy.bc_bidding_strategy import BcBiddingStrategy
from bidding_train_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
from bidding_train_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
from bidding_train_env.strategy.ppo_bidding_strategy import PpoBiddingStrategy
from bidding_train_env.strategy.onbc_transformer_bidding_strategy import ONBCTransformerBiddingStrategy


class BiddingStrategyFactory:
    strategy_dict = {
        "default": PlayerBiddingStrategy,
        "bc": BcBiddingStrategy,
        "onlineLp": OnlineLpBiddingStrategy,
        "iql": IqlBiddingStrategy,
        "ppo": PpoBiddingStrategy,
        "onbc_transformer": ONBCTransformerBiddingStrategy,
    }

    @classmethod
    def create(cls, strategy_name, **strategy_kwargs):
        strategy_class = cls.strategy_dict[strategy_name]
        return strategy_class(**strategy_kwargs)

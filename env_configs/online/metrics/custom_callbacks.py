import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

    
class TensorboardCallback(BaseCallback):
    def __init__(self, info_keywords, verbose=0):
        super().__init__(verbose=verbose)
        self.info_keywords = info_keywords
        self.rollout_info = {}
        
    def _on_rollout_start(self):
        self.rollout_info = {key: [] for key in self.info_keywords}
        
    def _on_step(self):
        for key in self.info_keywords:
            vals = [info.get(key) for info in self.locals["infos"] if info.get(key) is not None]
            self.rollout_info[key].extend(vals)
        return True
    
    def _on_rollout_end(self):
        for key in self.info_keywords:
            self.logger.record(key, np.mean(self.rollout_info[key]))


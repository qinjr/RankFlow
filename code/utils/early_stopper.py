from threading import Condition
import numpy as np

class EarlyStopper(object):
    def __init__(self, config):
        self.num_trials = config['num_trials']
        self.eval_metric_bigger = config['eval_metric_bigger']
        self.best_eval_score = -np.inf if self.eval_metric_bigger else np.inf
        self.trial_counter = 0
        self.best_epoch_idx = -1

    # return continue flag and save flag
    def check(self, eval_score, epoch_idx):
        condition = (eval_score > self.best_eval_score) if self.eval_metric_bigger else (eval_score < self.best_eval_score)
        if condition:
            self.best_eval_score = eval_score
            self.trial_counter = 0
            self.best_epoch_idx = epoch_idx
            return True, True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True, False
        else:
            return False, False
    
    def get_best_epoch_idx(self):
        return self.best_epoch_idx

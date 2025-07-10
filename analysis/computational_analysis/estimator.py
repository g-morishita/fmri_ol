import numpy as np
from scipy.optimize import minimize
from analysis.computational_analysis.nll import nll, nll_per_session_full

class FullModel:
    def __init__(self):
        self.reward_lr = None
        self.action_lr = None
        self.weight_for_A = None
        self.beta = None
    
    def reset(self):
        # Reset values to initial state
        self.reward_lr = None
        self.action_lr = None
        self.weight_for_A = None
        self.beta = None
    
    def fit(self, set_other_rewards, set_other_choices, set_self_choices):
        obj_func = lambda params: nll(params, nll_per_session_full, set_other_rewards, set_other_choices, set_self_choices)

        min_nll = float('inf')
        best_x = None

        for _ in range(30):
            x0 = [np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform(0, 100)]
            bounds = [(0, 1), (0, 1), (0, 1), (0, 100)]
            res = minimize(obj_func, x0, options={"maxiter": 1000, "disp": False}, bounds=bounds)
            if res.success:
                if min_nll > res.fun:
                    min_nll = res.fun
                    best_x = res.x
        self.reward_lr = best_x[0]
        self.action_lr = best_x[1]
        self.weight_for_A = best_x[2]
        self.beta = best_x[3]
        
        return best_x, min_nll

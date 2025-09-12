import numpy as np
from analysis.computational_analysis.nll import nll_per_session_full
from analysis.computational_analysis.simulator import  FullModel, AsocialRewardModel
import analysis.computational_analysis.estimator as estimator


class ParameterRecovery:
    def __init__(self, self_model, partner_model, reward_probs, n_trials=30, n_blocks=2):
        self.self_model = self_model
        self.partner_model = partner_model
        self.reward_probs = reward_probs
        self.n_blocks = n_blocks
        self.n_trials = n_trials
        self.set_partner_rewards = []
        self.set_partner_choices = []
        self.set_self_choices = []
     
    def simulate_block(self):
        for _ in range(self.n_blocks):
            is_repeated = True
            while is_repeated:
                self.self_model.reset()
                self.partner_model.reset()
                partner_rewasds = []
                partner_choices = []
                self_choices = []
                for _ in range(self.n_trials):
                    partner_choice = self.partner_model.make_choice()
                    partner_reward = int(np.random.uniform() < self.reward_probs[partner_choice])
                    self.partner_model.update_values(partner_reward, partner_choice)

                    partner_choices.append(partner_choice)
                    partner_rewasds.append(partner_reward)

                    self.self_model.update_values(partner_reward, partner_choice)
                    self_choice = self.self_model.make_choice()
                    self_choices.append(self_choice)
                
                # If the partner chose three different options, get out of the while loop
                is_repeated = len(set(partner_choices)) < 3
            
            self.set_partner_rewards.append(partner_rewasds)
            self.set_partner_choices.append(partner_choices)
            self.set_self_choices.append(self_choices)

    def fit_by_mle(self, estimator):
        # Fit the model to the simulated data
        params, nll = estimator.fit(self.set_partner_rewards, self.set_partner_choices, self.set_self_choices)
        return params, nll
    
    def fit_by_bayes(self):
        pass
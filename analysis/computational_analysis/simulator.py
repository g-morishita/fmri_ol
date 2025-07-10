import numpy as np

N_CHOICES = 3

class AsocialRewardModel:
    def __init__(self, reward_lr, beta):
        self.reward_lr = reward_lr
        self.beta = beta
        self.V = np.tile(0.5, N_CHOICES)
    
    def reset(self):
        # Reset values to initial state
        self.V = np.tile(0.5, N_CHOICES)
    
    def update_values(self, reward, self_cohice):
        # Update values according to observed choice and reward
        if self_cohice is not None:
            self.update_V(reward, self_cohice)
    
    def update_V(self, reward, self_choice):
        if self_choice is not None:
            self.V[self_choice] += self.reward_lr * (reward - self.V[self_choice])
    
    def get_choice_prob(self):
        exp_V = np.exp(self.beta * self.V)
        return exp_V / np.sum(exp_V)
    
    def make_choice(self):
        choice_prob = self.get_choice_prob()
        return np.random.choice(len(choice_prob), p=choice_prob)
    

class FullModel:
    def __init__(self, reward_lr, action_lr, weight_for_A, beta):
        self.reward_lr = reward_lr
        self.action_lr = action_lr
        self.weight_for_A = weight_for_A
        self.beta = beta
        self.V = np.tile(0.5, N_CHOICES)
        self.A = np.ones(3) / N_CHOICES
    
    def reset(self):
        # Reset values to initial state
        self.V = np.tile(0.5, N_CHOICES)
        self.A = np.ones(3) / N_CHOICES
    
    def update_values(self, reward, other_choice):
        # Update values according to observed choice and reward
        self.update_V(reward, other_choice)
        self.update_A(other_choice)
    
    def update_V(self, reward, other_choice):
        if other_choice is not None:
            self.V[other_choice] += self.reward_lr * (reward - self.V[other_choice])
        
    def update_A(self, other_choice):
        if other_choice is not None:
            self.A[other_choice] += self.action_lr * (1 - self.A[other_choice])
            for i in range(len(self.A)):
                if i != other_choice:
                    self.A[i] -= self.action_lr * self.A[other_choice]
    
    def get_choice_prob(self):
        Q = (1 - self.weight_for_A) * self.V + self.weight_for_A * self.A
        exp_Q = np.exp(self.beta * Q)
        return exp_Q / np.sum(exp_Q)
    
    def make_choice(self):
        choice_prob = self.get_choice_prob()
        return np.random.choice(len(choice_prob), p=choice_prob)
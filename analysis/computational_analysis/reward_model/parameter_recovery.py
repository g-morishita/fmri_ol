from analysis.computational_analysis.simulator import SocialRewardModel, AsocialRewardModel
from analysis.computational_analysis.estimator import MLERewardModel as EstimatorRewardModel
from analysis.computational_analysis.parameter_recovery import ParameterRecovery

def main():
    # Define the reward probabilities for the partner model
    reward_probs = [0.25, 0.5, 0.75]  # Example probabilities for each choice

    # Initialize the self and partner models
    self_model = SocialRewardModel(reward_lr=0.2, beta=3)
    partner_model = AsocialRewardModel(reward_lr=0.3, beta=1.5)

    # Create a parameter recovery instance
    parameter_recovery = ParameterRecovery(self_model, partner_model, reward_probs, n_trials=300, n_blocks=20)

    # Simulate blocks of choices
    parameter_recovery.simulate_block()

    # Fit the estimator to the simulated data
    estimator = EstimatorRewardModel()
    estimated_params, _ = parameter_recovery.fit(estimator)

    for idx, true_param in enumerate([self_model.reward_lr, self_model.action_lr, self_model.weight_for_A, self_model.beta]):
        print(f"True Parameter: {true_param}, Estimated: {estimated_params[idx]}")
 

if __name__ == "__main__":
    main()
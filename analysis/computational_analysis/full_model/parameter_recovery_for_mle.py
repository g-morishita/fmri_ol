from analysis.computational_analysis.simulator import FullModel, AsocialRewardModel
from analysis.computational_analysis.estimator import MLEFullModel as EstimatorFullModel
from analysis.computational_analysis.parameter_recovery import ParameterRecovery
import numpy as np
import sys


# Define the reward probabilities
REWARD_PROB = [0.25, 0.5, 0.75]


def sample_parameter():
    """Sample parameters for the FullModel."""
    reward_lr = np.random.uniform(0.01, 0.99)
    action_lr = np.random.uniform(0.01, 0.99)
    weight_for_A = np.random.uniform(0.01, 0.99)
    beta = 10 ** np.random.uniform(-1, 1)
    return reward_lr, action_lr, weight_for_A, beta


def perform_parameter_recovery():
    # Sample parameters for the FullModel
    reward_lr, action_lr, weight_for_A, beta = sample_parameter()

    # Initialize the self model with sampled parameters
    self_model = FullModel(reward_lr=reward_lr, action_lr=action_lr, weight_for_A=weight_for_A, beta=beta)
    partner_model = AsocialRewardModel(reward_lr=0.3, beta=1.5)

    # Create a parameter recovery instance
    parameter_recovery = ParameterRecovery(self_model, partner_model, REWARD_PROB, n_trials=30, n_blocks=2)
    # Simulate blocks of choices
    parameter_recovery.simulate_block()
    # Fit the estimator to the simulated data
    estimator = EstimatorFullModel()
    estimated_params, _ = parameter_recovery.fit(estimator)
    
    return estimated_params, (reward_lr, action_lr, weight_for_A, beta)


def main(n_simulations):
    results = {"true_learning_rate_for_reward": [],
               "true_learning_rate_for_action": [],
               "true_relative_weight_of_action": [],
               "true_beta": [],
               "estimated_learning_rate_for_reward": [],
               "estimated_learning_rate_for_action": [],
               "estimated_relative_weight_of_action": [],
               "estimated_beta": []}
    for i in range(n_simulations):
        print(f"Simulation {i + 1}/{n_simulations}")
        estimated_params, true_params = perform_parameter_recovery()
        
        for estimated, true, key in zip(estimated_params, true_params,
                                   ["learning_rate_for_reward", "learning_rate_for_action",
                                    "relative_weight_of_action", "beta"]):
            results[f"estimated_{key}"].append(estimated)
            results[f"true_{key}"].append(true)
            # Print the results
            print(f"True {key}: {true}, Estimated: {estimated}")
        print("\n")

    # Save the parameter recovery result to csv file
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results/computational_analysis/full_model_parameter_recovery_results.csv", index=False)

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    if len(sys.argv) > 1:
    # Read number of simulations from command line argument
        n_simulations = int(sys.argv[1])
    else:
        print("Usage: python parameter_recovery.py <n_simulations>")
    n_simulations = int(sys.argv[1])
    main(n_simulations)
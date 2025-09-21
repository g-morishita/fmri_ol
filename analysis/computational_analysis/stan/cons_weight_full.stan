data {
  int<lower=1> n_subjects;
  int<lower=1> n_blocks;
  int<lower=1> n_trials;
  int<lower=1> n_choices;

  array[n_subjects, n_blocks, n_trials] int self_choices;    // 1..n_choices, or <=0 if missing
  array[n_subjects, n_blocks, n_trials] int other_choices;   // 1..n_choices, or <=0 if missing
  array[n_subjects, n_blocks, n_trials] int other_rewards;   // typically 0/1
  array[n_subjects, n_blocks] int<lower=1, upper=2> noise_level_condition; // 1=low, 2=high
}

parameters {
  // population-level (latent scale) for baseline (low-noise condition)
  real mu_latent_lr_reward_low_noise;
  real<lower=0> sigma_latent_lr_reward_low_noise;

  real mu_latent_lr_action_low_noise;
  real<lower=0> sigma_latent_lr_action_low_noise;

  real mu_latent_weight_for_A_low_noise;
  real<lower=0> sigma_latent_weight_for_A_low_noise;

  real mu_latent_beta_low_noise;
  real<lower=0> sigma_latent_beta_low_noise;

  // population-level (latent scale) for difference between conditions (high - low)
  real mu_delta_latent_lr_reward;
  real<lower=0> sigma_delta_latent_lr_reward;

  real mu_delta_latent_lr_action;
  real<lower=0> sigma_delta_latent_lr_action;

  real mu_delta_latent_weight_for_A;
  real<lower=0> sigma_delta_latent_weight_for_A;

  real mu_delta_latent_beta;
  real<lower=0> sigma_delta_latent_beta;

  // standard normal subject deviations (non-centered)
  vector[n_subjects] normal_lr_reward;
  vector[n_subjects] normal_delta_lr_reward;

  vector[n_subjects] normal_lr_action;
  vector[n_subjects] normal_delta_lr_action;

  vector[n_subjects] normal_weight_for_A;
  vector[n_subjects] normal_delta_weight_for_A;

  vector[n_subjects] normal_beta;
  vector[n_subjects] normal_delta_beta;
}

transformed parameters {
  // subject-level latent parameters (non-centered construction)
  vector[n_subjects] latent_lr_reward_low_noise;
  vector[n_subjects] latent_lr_reward_high_noise;

  vector[n_subjects] latent_lr_action_low_noise;
  vector[n_subjects] latent_lr_action_high_noise;

  vector[n_subjects] latent_weight_for_A_low_noise;
  vector[n_subjects] latent_weight_for_A_high_noise;

  vector[n_subjects] latent_beta_low_noise;
  vector[n_subjects] latent_beta_high_noise;

  // transformed per subject x condition (on constrained scales)
  vector<lower=0,upper=1>[n_subjects] lr_reward_low_noise;
  vector<lower=0,upper=1>[n_subjects] lr_reward_high_noise;

  vector<lower=0,upper=1>[n_subjects] lr_action_low_noise;
  vector<lower=0,upper=1>[n_subjects] lr_action_high_noise;

  vector<lower=0,upper=1>[n_subjects] weight_for_A_low_noise;
  vector<lower=0,upper=1>[n_subjects] weight_for_A_high_noise;

  vector<lower=0>[n_subjects] beta_low_noise;
  vector<lower=0>[n_subjects] beta_high_noise;

  // build latent subjects
  for (i in 1:n_subjects) {
    latent_lr_reward_low_noise[i]  = mu_latent_lr_reward_low_noise
                                   + sigma_latent_lr_reward_low_noise  * normal_lr_reward[i];
    latent_lr_reward_high_noise[i] = latent_lr_reward_low_noise[i]
                                   + mu_delta_latent_lr_reward
                                   + sigma_delta_latent_lr_reward * normal_delta_lr_reward[i];

    latent_lr_action_low_noise[i]  = mu_latent_lr_action_low_noise
                                   + sigma_latent_lr_action_low_noise  * normal_lr_action[i];
    latent_lr_action_high_noise[i] = latent_lr_action_low_noise[i]
                                   + mu_delta_latent_lr_action
                                   + sigma_delta_latent_lr_action * normal_delta_lr_action[i];

    latent_weight_for_A_low_noise[i]  = mu_latent_weight_for_A_low_noise
                                      + sigma_latent_weight_for_A_low_noise * normal_weight_for_A[i];
    latent_weight_for_A_high_noise[i] = latent_weight_for_A_low_noise[i]
                                      + mu_delta_latent_weight_for_A
                                      + sigma_delta_latent_weight_for_A * normal_delta_weight_for_A[i];

    latent_beta_low_noise[i]  = mu_latent_beta_low_noise
                              + sigma_latent_beta_low_noise * normal_beta[i];
    latent_beta_high_noise[i] = latent_beta_low_noise[i]
                              + mu_delta_latent_beta
                              + sigma_delta_latent_beta * normal_delta_beta[i];

    // map to constrained scales
    lr_reward_low_noise[i]     = Phi(latent_lr_reward_low_noise[i]);
    lr_reward_high_noise[i]    = Phi(latent_lr_reward_high_noise[i]);

    lr_action_low_noise[i]     = Phi(latent_lr_action_low_noise[i]);
    lr_action_high_noise[i]    = Phi(latent_lr_action_high_noise[i]);

    weight_for_A_low_noise[i]  = Phi(latent_weight_for_A_low_noise[i]);
    weight_for_A_high_noise[i] = Phi(latent_weight_for_A_high_noise[i]);

    beta_low_noise[i]          = exp(latent_beta_low_noise[i]);
    beta_high_noise[i]         = exp(latent_beta_high_noise[i]);
  }
}

model {
  // Priors for population-level parameters
  mu_latent_lr_reward_low_noise       ~ normal(0, 0.5);
  sigma_latent_lr_reward_low_noise    ~ normal(0, 1);

  mu_latent_lr_action_low_noise       ~ normal(0, 0.5);
  sigma_latent_lr_action_low_noise    ~ normal(0, 1);

  mu_latent_weight_for_A_low_noise    ~ normal(0, 0.5);
  sigma_latent_weight_for_A_low_noise ~ normal(0, 1);

  mu_latent_beta_low_noise            ~ normal(0, 0.5);
  sigma_latent_beta_low_noise         ~ normal(0, 1);

  mu_delta_latent_lr_reward           ~ normal(0, 0.5);
  sigma_delta_latent_lr_reward        ~ normal(0, 1);

  mu_delta_latent_lr_action           ~ normal(0, 0.5);
  sigma_delta_latent_lr_action        ~ normal(0, 1);

  mu_delta_latent_weight_for_A        ~ normal(0, 0.5);
  sigma_delta_latent_weight_for_A     ~ normal(0, 1);

  mu_delta_latent_beta                ~ normal(0, 0.5);
  sigma_delta_latent_beta             ~ normal(0, 1);

  // Standard normals for non-centered parameterization
  normal_lr_reward            ~ normal(0, 1);
  normal_delta_lr_reward      ~ normal(0, 1);

  normal_lr_action            ~ normal(0, 1);
  normal_delta_lr_action      ~ normal(0, 1);

  normal_weight_for_A         ~ normal(0, 1);
  normal_delta_weight_for_A   ~ normal(0, 1);

  normal_beta                 ~ normal(0, 1);
  normal_delta_beta           ~ normal(0, 1);

  // Likelihood (inline updates)
  {
    vector[n_choices] values;
    vector[n_choices] tendencies;
    vector[n_choices] combined;

    for (i in 1:n_subjects) {
      for (b in 1:n_blocks) {
        // reset per block
        values     = rep_vector(0.5, n_choices);
        tendencies = rep_vector(1.0 / n_choices, n_choices);  // was 1/3, now general

        int  cond   = noise_level_condition[i, b]; // 1=low, 2=high
        real lr_r   = lr_reward_high_noise[i];
        real lr_a   = lr_action_high_noise[i];
        real wA     = weight_for_A_high_noise[i];
        real beta_i = beta_high_noise[i];

        if (cond == 1) { // if condition is low-noise
          lr_r   = lr_reward_low_noise[i];
          lr_a   = lr_action_low_noise[i];
          wA     = weight_for_A_low_noise[i];
          beta_i = beta_low_noise[i];
        }

        for (t in 1:n_trials) {
          int oc = other_choices[i, b, t];
          int rw = other_rewards[i, b, t];
          int sc = self_choices[i, b, t];

          // update from other's observation if valid
          if (oc > 0) {
            tendencies -= lr_a * tendencies;
            tendencies[oc] += lr_a;

            values[oc] += lr_r * (rw - values[oc]);
          }

          // log-likelihood if self choice is valid
          if (sc > 0) {
            combined = wA * tendencies + (1 - wA) * values;
            target += categorical_logit_lpmf(sc | beta_i * combined);
          }
        }
      }
    }
  }
}

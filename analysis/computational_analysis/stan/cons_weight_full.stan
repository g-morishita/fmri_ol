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
  // population-level (latent scale) per condition: [1]=low, [2]=high
  vector[2] mu_latent_lr_reward;
  vector<lower=0>[2] sigma_latent_lr_reward;

  vector[2] mu_latent_lr_action;
  vector<lower=0>[2] sigma_latent_lr_action;

  vector[2] mu_latent_weight_for_A;
  vector<lower=0>[2] sigma_latent_weight_for_A;

  vector[2] mu_latent_beta;
  vector<lower=0>[2] sigma_latent_beta;

  // subject-level (latent scale) per condition
  matrix[n_subjects, 2] latent_lr_reward;
  matrix[n_subjects, 2] latent_lr_action;
  matrix[n_subjects, 2] latent_weight_for_A;
  matrix[n_subjects, 2] latent_beta;
}

transformed parameters {
  // transformed per subject x condition
  matrix<lower=0,upper=1>[n_subjects, 2] lr_reward;
  matrix<lower=0,upper=1>[n_subjects, 2] lr_action;
  matrix<lower=0,upper=1>[n_subjects, 2] weight_for_A;
  matrix<lower=0>[n_subjects, 2] beta;

  for (i in 1:n_subjects) {
    for (c in 1:2) {
      lr_reward[i, c]      = Phi(mu_latent_lr_reward[c]      + sigma_latent_lr_reward[c]      * latent_lr_reward[i, c]);
      lr_action[i, c]      = Phi(mu_latent_lr_action[c]      + sigma_latent_lr_action[c]      * latent_lr_action[i, c]);
      weight_for_A[i, c]   = Phi(mu_latent_weight_for_A[c]   + sigma_latent_weight_for_A[c]   * latent_weight_for_A[i, c]);
      beta[i, c]           = exp(mu_latent_beta[c] + sigma_latent_beta[c] * latent_beta[i, c]);
    }
  }
}

model {
  // priors
  mu_latent_lr_reward     ~ normal(0, 1);
  sigma_latent_lr_reward  ~ normal(0, 1);     // half-normal via <lower=0>

  mu_latent_lr_action     ~ normal(0, 1);
  sigma_latent_lr_action  ~ normal(0, 1);     // half-normal via <lower=0>

  mu_latent_weight_for_A  ~ normal(0, 1);
  sigma_latent_weight_for_A ~ normal(0, 1);   // half-normal via <lower=0>

  mu_latent_beta          ~ normal(0, 1);
  sigma_latent_beta       ~ normal(0, 1);     // half-normal via <lower=0>

  to_vector(latent_lr_reward)     ~ normal(0, 1);
  to_vector(latent_lr_action)     ~ normal(0, 1);
  to_vector(latent_weight_for_A)  ~ normal(0, 1);
  to_vector(latent_beta)          ~ normal(0, 1);

  // likelihood (inline updates)
  {
    vector[n_choices] values;
    vector[n_choices] tendencies;
    vector[n_choices] combined;

    for (i in 1:n_subjects) {
      for (b in 1:n_blocks) {
        // reset per block
        values     = rep_vector(0.5, n_choices);
        tendencies = rep_vector(1.0 / 3.0, n_choices);

        int  cond   = noise_level_condition[i, b]; // 1=low, 2=high
        real lr_r   = lr_reward[i, cond];
        real lr_a   = lr_action[i, cond];
        real wA     = weight_for_A[i, cond];
        real beta_i = beta[i, cond];

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

generated quantities {
  // Pointwise log-likelihood for WAIC/LOO (zeros for missing choices)
  array[n_subjects, n_blocks, n_trials] real log_lik;

  {
    vector[n_choices] values;
    vector[n_choices] tendencies;
    vector[n_choices] combined;

    for (i in 1:n_subjects) {
      for (b in 1:n_blocks) {
        // reset per block
        values     = rep_vector(0.5, n_choices);
        tendencies = rep_vector(1.0 / 3.0, n_choices);

        int  cond   = noise_level_condition[i, b];
        real lr_r   = lr_reward[i, cond];
        real lr_a   = lr_action[i, cond];
        real wA     = weight_for_A[i, cond];
        real beta_i = beta[i, cond];

        for (t in 1:n_trials) {
          int oc = other_choices[i, b, t];
          int rw = other_rewards[i, b, t];
          int sc = self_choices[i, b, t];

          // same state updates as in model
          if (oc > 0) {
            tendencies -= lr_a * tendencies;
            tendencies[oc] += lr_a;

            values[oc] += lr_r * (rw - values[oc]);
          }

          // store pointwise log-lik or 0 if missing
          if (sc > 0) {
            combined = wA * tendencies + (1 - wA) * values;
            log_lik[i, b, t] = categorical_logit_lpmf(sc | beta_i * combined);
          } else {
            log_lik[i, b, t] = 0;
          }
        }
      }
    }
  }
}

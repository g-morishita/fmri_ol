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
  // population (latent scale) per condition: [1]=low, [2]=high
  vector[2] mu_latent_lr;
  vector<lower=0>[2] sigma_latent_lr;

  vector[2] mu_latent_beta;
  vector<lower=0>[2] sigma_latent_beta;

  // subject-level latents (standard normal) per condition
  matrix[n_subjects, 2] latent_lr;
  matrix[n_subjects, 2] latent_beta;
}

transformed parameters {
  // transformed, per subject × condition
  matrix<lower=0, upper=1>[n_subjects, 2] lr;
  matrix<lower=0>[n_subjects, 2]         beta;

  for (i in 1:n_subjects) {
    for (c in 1:2) {
      lr[i, c]   = Phi( mu_latent_lr[c]   + sigma_latent_lr[c]   * latent_lr[i, c] );
      beta[i, c] = exp( mu_latent_beta[c] + sigma_latent_beta[c] * latent_beta[i, c] );
    }
  }
}

model {
  // priors
  mu_latent_lr     ~ normal(0, 1);
  sigma_latent_lr  ~ normal(0, 1);     // half-normal via <lower=0>
  mu_latent_beta   ~ normal(0, 1);
  sigma_latent_beta~ normal(0, 1);     // half-normal via <lower=0>
  to_vector(latent_lr)   ~ normal(0, 1);
  to_vector(latent_beta) ~ normal(0, 1);

  // likelihood
  {
    vector[n_choices] values;
    int observed_choice;
    int observed_reward;
    int self_choice;

    for (i in 1:n_subjects) {
      for (b in 1:n_blocks) {
        values = rep_vector(1/3, n_choices); // reset Q-values each block
        int cond = noise_level_condition[i, b];    // 1=low, 2=high
        real lr_i   = lr[i, cond];
        real beta_i = beta[i, cond];

        for (t in 1:n_trials) {
          observed_choice = other_choices[i, b, t];
          observed_reward = other_rewards[i, b, t];
          self_choice     = self_choices[i, b, t];

          // TD update only if we have a valid observed_action
          if (observed_choice > 0) {
            values -= lr_i * values;
            values[observed_choice] += lr_i;
          }

          // choice likelihood only if the self action is valid
          if (self_choice > 0)
            target += categorical_logit_lpmf(self_choice | beta_i * values);
            // equivalently: target += (beta_i * values)[self_choice] - log_sum_exp(beta_i * values);
        }
      }
    }
  }
}

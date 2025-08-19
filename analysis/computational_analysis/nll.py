import numpy as np
from scipy.special import softmax


def _skip_choice(sc):
    return sc is None or (isinstance(sc, float) and np.isnan(sc))


def update_V(V, r, c, lr):
    delta = r - V[c]
    V[c] += lr * delta


def update_A(A, c, lr):
    # for chosen action
    delta = 1 - A[c]
    A[c] += lr * delta

    # For unchosen actions
    for a in range(3):
        if a != c:
            delta = 0 - A[a]
            A[a] += lr * delta


def get_choice_prob(Q, beta):
    p = softmax(beta * Q)
    return p


def nll(params, nll_per_session, set_other_rewards, set_other_choices, set_self_choices):
    n_sessions = len(set_other_rewards)

    nll = 0
    for i in range(n_sessions):
        nll += nll_per_session(params, set_other_rewards[i], set_other_choices[i], set_self_choices[i])

    return nll


def nll_per_session_full(params, other_rewards, other_choices, self_choices):
    reward_lr, action_lr, weight_for_A, beta = params
    V = np.tile(0.5, 3)
    A = np.ones(3) / 3

    nll = 0
    for r, oc, sc in zip(other_rewards, other_choices, self_choices):
        # Update values according to observed choice and reward
        update_V(V, r, oc, reward_lr)
        update_A(A, oc, action_lr)

        # Ignore cases where self choice is None (mishit)
        if _skip_choice(sc):
            continue

        # Calculate choice probability
        Q = (1 - weight_for_A) * V + weight_for_A * A
        choice_prob = get_choice_prob(Q, beta)
        nll -= np.log(choice_prob[int(sc)] + 1e-10)  # Adding a small value to avoid log(0)

    return nll


def nll_per_session_q(params, other_rewards, other_choices, self_choices):
    reward_lr, beta = params
    V = np.tile(0.5, 3)

    nll = 0
    for r, oc, sc in zip(other_rewards, other_choices, self_choices):
        # Update values according to observed choice and reward
        update_V(V, r, oc, reward_lr)

        # Ignore cases where self choice is None (mishit)
        if _skip_choice(sc):
            continue
        # Calculate choice probability
        choice_prob = get_choice_prob(V, beta)
        nll -= np.log(choice_prob[sc] + 1e-10)  # Adding a small value to avoid log(0)

    return nll


def nll_per_session_action(params, other_rewards, other_choices, self_choices):
    action_lr, beta = params
    A = np.ones(3) / 3

    nll = 0
    for r, oc, sc in zip(other_rewards, other_choices, self_choices):
        # Update values according to observed choice and reward
        update_A(A, oc, action_lr)

        # Ignore cases where self choice is None (mishit)
        if _skip_choice(sc):
            continue
        # Calculate choice probability
        choice_prob = get_choice_prob(A, beta)
        nll -= np.log(choice_prob[sc] + 1e-10)  # Adding a small value to avoid log(0)

    return nll
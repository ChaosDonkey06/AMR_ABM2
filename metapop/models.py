import pandas as pd
import numpy as np

#def binomial_transition(xi, tau, m, dt=1):
#    kb    = np.maximum(1.0 - np.exp(-tau*dt), 0)
#    pop   = np.random.binomial(list(xi), kb, m)
#    return pop
def binomial_transition(n, prob):
    return np.random.binomial(n, prob)

def poisson_transition(n, rate):
    return np.random.poisson(np.nan_to_num(n * rate))

def deterministic_transition(n, rate):
    return np.round(np.nan_to_num(n * rate), 0)

def check_state_space(x, pop=None):
    return np.clip(x, 0, pop)

def process_metapop(t, x, gamma, beta, delta, Nmean, N, A, D, M, model_settings=None):

    """ Susceptible - Colonized meta-population model

    Args:
        x[t]  : state space
        gamma : importation rate
        beta  : transmission rate
        delta : decolonization rate
        A     : Admitted
        D     : Discharged
        M     : Movements matrix

    Returns:
        x[t+1]: State space in t+1.
    """
    C = x[0, :, :]
    S = np.clip(N - C,0,N)

    c = np.clip(np.nan_to_num(C/N), 0, 1)

    λ = beta * C / Nmean # force of infection

    # moving out and in colonized
    Cout  = binomial_transition(list(np.sum(M, axis=1, keepdims=True)), c)
    Cin   = M.T @ c

    a2c   = binomial_transition(list(A), gamma) # people admitted colonized.
    c2d   = binomial_transition(list(D), c)     # discharged colonized

    s2c  = poisson_transition(S, λ)     # new colonized
    c2s  = poisson_transition(C, delta) # decolonizations

    C    = C + a2c - c2d + s2c + c2s + Cin - Cout
    C    = np.clip(C, 0, N)

    return check_state_space(np.array([C, a2c, s2c]))

def observe_metapop(t, x, N, rho, num_tests, model_settings):
    """ Observational model
        Args:
            t (int):      Time
            x (np.array): State space
            rho (float):  Observation probability
        Returns:
            y (np.array): Observed carriers ~ Binomial(C, rho)
    """

    m       = model_settings["m"]
    num_pop = model_settings["num_pop"]
    C       = x[0, :, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        observed_colonized = np.random.binomial(list(num_tests * np.ones((num_pop, m))), rho * np.clip(np.nan_to_num(C/N), 0, 1))

    return observed_colonized

def init_metapop(N0, c0, model_settings):
    """ Initial conditions model.
        Args:
            N0 (int):    Initial size of populations.
            c0 (int):    Initial fraction of carriers.
        Returns:
            x0 (np.array): Initial conditions of the state space.
    """
    m       = model_settings["m"]
    num_pop = model_settings["num_pop"]

    N0 = np.expand_dims(N0, -1) * np.ones((num_pop, m))
    C0   = c0 * N0
    AC   = np.zeros((num_pop, m))
    newC = np.zeros((num_pop, m))

    return np.array([C0, AC, newC])

def simulate_metapop(process_model, observational_model, init_state, θsim, model_settings):

    """ Simulate model with initial conditions and parameters
        x \in R^{n/num_pop x num_pop x ms}

    Args:
        model (function):        Process model
        observe (function):      Observational model
        initial_x0 (function):   Initial condition guess model.
        θ_sim (np.array):        Parameters
    """
    n = model_settings["n"]
    k = model_settings["k"]
    m = model_settings["m"]
    T = model_settings["T"]
    num_pop = model_settings["num_pop"]

    x0 = init_state(θsim)

    if(x0.shape[0] != n/num_pop or x0.shape[2] != m or x0.shape[1] != num_pop) :
        print('error in x0 dimensions')

    x_sim = np.full((T, n/num_pop, num_pop, m), np.nan)
    y_sim = np.full((T, k, m), np.nan)

    x_sim[0, :, :, :] = x0
    y_sim[0, :, :]    = observational_model(0, x0, θsim)
    for t in range(1, T-1):
        x_sim[t, :, :, :] = process_model(t, x_sim[t-1, :, :, :], θsim)
        y_sim[t, :, :]    = observational_model(t, x_sim[t, :, :, :], θsim)

    return x_sim, y_sim
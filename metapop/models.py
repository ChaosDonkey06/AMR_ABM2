import pandas as pd
import numpy as np

#def binomial_transition(xi, tau, m, dt=1):
#    kb    = np.maximum(1.0 - np.exp(-tau*dt), 0)
#    pop   = np.random.binomial(list(xi), kb, m)
#    return pop

def binomial_transition(n, prob, m):
    return np.random.binomial(n, prob, m)

def poisson_transition(n, rate, m):
    return np.random.poisson(np.nan_to_num(n * rate), m)

def deterministic_transition(n, rate):
    return np.round(snp.nan_to_num(n * rate), 0)

def check_state_space(x, pop=None):
    return np.clip(x, 0, pop)

def process_metapop(t, x, gamma, beta, delta, Nmean, N, A, D, M, model_settings):
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

    num_pop    = model_settings["num_pop"]
    m          = model_settings["m"]
    stochastic = model_settings["stochastic"]

    C = x[range(0,num_pop), :]
    N = np.reshape([N]*m,(m,num_pop)).T
    S = np.clip(N - C,0,N)

    transition_b = lambda n, prob, m: binomial_transition(n, prob, m) if stochastic else deterministic_transition(n, prob)
    transition_p = lambda n, rate, m: poisson_transition(n, rate, m) if stochastic else deterministic_transition(n, rate)

    with np.errstate(divide='ignore', invalid='ignore'):

        moveitojC = np.full((num_pop,num_pop,m),0)
        for j in range(0, num_pop):
            for i in range(0,num_pop):
                if(i!=j):
                    moveitojC[i,j,:] = transition_b(M[i,j], np.clip(np.nan_to_num(C[i, :] / N[i, :]),0,1), m)

        AC       = np.zeros((num_pop, m))
        DC       = np.zeros((num_pop, m))
        newC     = np.zeros((num_pop, m))
        leftC    = np.zeros((num_pop, m))
        moveinC  = np.zeros((num_pop, m))
        moveoutC = np.zeros((num_pop, m))

        for j in range(0,num_pop) :
            AC[j, :]      = transition_b(A[j], gamma, m)                                         # admitted colonized
            DC[j,:]       = transition_b(D[j], np.clip(np.nan_to_num(C[j, :]/N[j, :]),0,1), m)   # discharged colonized

            newC[j,:]     = transition_p(S[j, :], np.nan_to_num(beta[j, :]*C[j, :]/Nmean[j]), m) # new colonized
            leftC[j,:]    = transition_p(C[j, :], delta, m)

            moveinC[j,:]  = np.sum(moveitojC[:,j,:], 0)
            moveoutC[j,:] = np.sum(moveitojC[j,:,:], 0)

            C[j, :]       = C[j, :] + AC[j,:] - DC[j,:] + newC[j,:] - leftC[j,:] + moveinC[j,:] - moveoutC[j,:]
            C[j, :]       = np.clip(C[j, :], 0, N[j,:])

    return check_state_space(np.concatenate((C,AC,newC)))

def observe_metapop(t, x, N, rho, tests, model_settings):
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
    C       = x[range(0,num_pop), :]
    N       = np.reshape([N]*m,(m,num_pop)).T
    tests   = np.reshape(list(tests)*m,(m,len(tests))).T

    with np.errstate(divide='ignore', invalid='ignore'):
        observed_colonized = np.random.binomial(tests.astype('int'), rho * np.nan_to_num(C/N))

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

    N  = np.reshape([N0] * m, (m, num_pop)).T
    c0 = np.reshape([c0] * num_pop, (m, num_pop)).T
    C  = c0*N

    AC   = np.zeros((num_pop, m))
    newC = np.zeros((num_pop, m))

    return np.concatenate((C, AC, newC))


def simulate(model, observe, initial_x0, θsim, model_settings):
    """ Simulate model with initial conditions and parameters

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

    x0 = initial_x0(θsim)
    if(x0.shape[0] != n or x0.shape[1] != m) :
        print('error in x0 dimensions')

    x_sim = np.full((n, m, T), np.nan)
    y_sim = np.full((k, m, T), np.nan)

    x_sim[:, :, 0] = x0
    y_sim[:, :, 0] = observe(0, x0, θsim)
    for t in range(1, T):
        x_sim[:, :, t] = model(t, x_sim[:, :, t-1], θsim)
        y_sim[:, :, t] = observe(t, x_sim[:, :, t], θsim)

    return x_sim, y_sim


import numpy as np

class Patient:
    susceptible = 0
    colonized   = 1

class Observed:
    no  = 0
    yes = 1

def amr_abm(t, agents_state, gamma, beta, alpha, movement, ward2size, model_settings):
    """ Agent based model tracking colonized and susceptible patients with pre-defined movement patterns.

    Args:
        agents_state : agent state. {0: Patient.susceptible, 1: Patient.colonized}  Size: (n_patients)
        movement     : pd.Dataframe with patient locations and culture information.
        parameters   : dictionary of parameters, contains importation rate (gamma), nosocomial transmission rate (beta),
                        effective sensitivity (ro), and decolonization rate (alpha)
    """

    n  = model_settings["n"] # number of patients
    m  = model_settings["m"] # number of ensembles

    p_update = agents_state.copy()
    p_update = Patient.susceptible * (agents_state * np.random.random(size=(n, m)) <= alpha)

    new_patients = movement[movement["first_day"]==1]["mrn_id"].values
    if new_patients.shape[0] > 0:
        p_update[new_patients, :] = Patient.colonized * (np.random.random(size=(new_patients.shape[0], m)) <= gamma)

    for i, ward_id in enumerate(movement["ward_id"].unique()):
        patients_ward = movement[movement["ward_id"]==ward_id]["mrn_id"].values
        λ_i = beta * np.sum(p_update[patients_ward, :]==Patient.colonized) / ward2size[ward_id]
        p_update[patients_ward, :] = p_update[patients_ward, :] + Patient.colonized * (np.random.random(size=(patients_ward.shape[0], m)) <= λ_i)
    p_update = np.clip(p_update, 0, 1)
    return p_update

def observe_cluster(t, agents_state, movement, rho, model_settings):
    ρ      = rho                  # effective sensitivity.
    k      = model_settings["k"]  # number of observations / clusters / buildings
    m      = model_settings["m"]  # number of clusters

    cluster_positive  = np.zeros((k, m))
    p_test            = Observed.yes * (np.random.random(size=(agents_state.shape[0], m)) <= agents_state * ρ)
    for i, cluster in enumerate(movement["cluster"].unique()):
        patients_test_ward            = movement.query(f"cluster=={cluster} and test==True")["mrn_id"].values
        cluster_positive[cluster,  :] = np.sum(p_test[patients_test_ward, :]    == Observed.yes, axis=0)

    return cluster_positive

def f0(model_settings):
    """ Initial state of the model.
    """
    n             = model_settings["n"]  # number of patients / size of the state space.
    m             = model_settings["m"]  # number of ensembles.
    patient_state = np.zeros((n, m))
    return patient_state
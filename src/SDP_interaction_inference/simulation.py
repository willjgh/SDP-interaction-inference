'''
Module to simulate synthetic data for testing inference methods.

Simulate stochastic reaction network models using the gillespie algorithm to
produce stationary samples emulating single-cell RNA sequencing observations.
Simulate datatsets of samples from large numbers of genes by sampling model
parameters from distributions fitted to match observed data.

Typical example:

# create generator with fixed seed
rng = np.random.default_rng(200)

# set model parameters
params = {
    'k_tx_1': 1,
    'k_tx_2': 1,
    'k_deg_1': 1,
    'k_deg_2': 1,
    'k_reg': 1
}

# simulate sample
sample = gillespie(rng, params, 1000, 0.5)
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_interaction_inference import dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

# ------------------------------------------------
# Gillespie simulation
# ------------------------------------------------

def gillespie_telegraph(params, n, tmax=100, ts=10, plot=False, initial_state=(0, 0, 1, 1)):
    '''
    Simulate a sample path of telegraph regulation model.

    Gillespie algorithm to simulate a sample path of the markov chain described by the
    telegraph regulation stochastic reaction network model with given parameters. After a
    burn-in time of 'tmax' samples are taken from the sample path at time intervals of 'ts'.
    The states / samples are pairs of counts (x1, x2) from a pair of genes.

    Args:

    Returns:
    '''

    # initialize random generator
    rng = np.random.default_rng()

    # initialise time and state
    t = 0
    path = [initial_state]
    jump_times = [0]

    # simulate for burn-in time and time between n samples
    while t < tmax + (n - 1) * ts:

        # current state
        x1, x2, g1, g2 = path[-1][0], path[-1][1], path[-1][2], path[-1][3]

        # transition rates
        q_on_1 = (1 - g1) * params['k_on_1']
        q_on_2 = (1 - g2) * params['k_on_2']
        q_off_1 = g1 * params['k_off_1']
        q_off_2 = g2 * params['k_off_2']
        q_tx_1 = g1 * params['k_tx_1']
        q_tx_2 = g2 * params['k_tx_2']
        q_deg_1 = x1 * params['k_deg_1']
        q_deg_2 = x2 * params['k_deg_2']
        q_reg = x1 * x2 * params['k_reg']
        q_hold = q_on_1 + q_on_2 + q_off_1 + q_off_2 + q_tx_1 + q_tx_2 + \
            q_deg_1 + q_deg_2 + q_reg

        # holding time in current state
        t_hold = -np.log(rng.uniform()) / q_hold
        t += t_hold
        jump_times.append(t)

        # jump probability
        outcome = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        prob = [
            q_on_1 / q_hold,
            q_on_2 / q_hold,
            q_off_1 / q_hold,
            q_off_2 / q_hold,
            q_tx_1 / q_hold,
            q_tx_2 / q_hold,
            q_deg_1 / q_hold,
            q_deg_2 / q_hold,
            q_reg / q_hold
        ]
        jump = rng.choice(outcome, p=prob)
        match jump:
            case 1:
                path.append((x1, x2, g1 + 1, g2))
            case 2:
                path.append((x1, x2, g1, g2 + 1))
            case 3:
                path.append((x1, x2, g1 - 1, g2))
            case 4:
                path.append((x1, x2, g1, g2 - 1))
            case 5:
                path.append((x1 + 1, x2, g1, g2))
            case 6:
                path.append((x1, x2 + 1, g1, g2))
            case 7:
                path.append((x1 - 1, x2, g1, g2))
            case 8:
                path.append((x1, x2 - 1, g1, g2))
            case 9:
                path.append((x1 - 1, x2 - 1, g1, g2))

    # take the transcript states
    x1_path = [state[0] for state in path]
    x2_path = [state[1] for state in path]

    # create step function of sample path from jump times and jump values
    x1_path_function = scipy.interpolate.interp1d(jump_times, x1_path, kind='previous')
    x2_path_function = scipy.interpolate.interp1d(jump_times, x2_path, kind='previous')

    # take values at sampling times as samples from stationary dist
    sample_times = [tmax + i * ts for i in range(n)]
    x1_samples = x1_path_function(sample_times)
    x2_samples = x2_path_function(sample_times)

    # convert to integers
    x1_samples = [int(x1) for x1 in x1_samples]
    x2_samples = [int(x2) for x2 in x2_samples]

    # re-combine to pairs of samples
    samples = list(zip(x1_samples, x2_samples))

    # plot sample paths
    if plot:
        x = np.linspace(0, tmax + (n - 1) * ts, 10000)
        plt.plot(x, x1_path_function(x), label="X1 sample path", color="blue")
        plt.plot(x, x2_path_function(x), label="X2 sample path", color="purple")
        #plt.axvline(tmax, label="Burn-in time", color="orange")
        plt.xlabel("Time")
        plt.ylabel("# molcules")
        plt.legend()
        plt.show()

    return samples

def gillespie_birth_death(params, n, tmax=100, ts=10, plot=False, initial_state=(0, 0)):
    '''
    Simulate a sample path of birth-death regulation model.

    Gillespie algorithm to simulate a sample path of the markov chain described
    by the birth-death regulation stochastic reaction network model with given
    parameters. After a burn-in time of 'tmax' samples are taken from the sample
    path at time intervals of 'ts'. The states / samples are pairs of counts
    (x1, x2) from a pair of genes.

    Args:
        params: dict of reaction rate constants 'k_tx_1', 'k_tx_2', 'k_deg_1',
                'k_deg_2', 'k_deg'
        n: sample size
        tmax: burn-in time of simulation
        ts: time between samples
        plot: toggle plotting of sample path
        intitial_state: starting state of simulation

    Returns:
        samples: n pairs of integers sampled from the reaction network
    '''

    # initialize random generator
    rng = np.random.default_rng()

    # initialise time and state
    t = 0
    path = [initial_state]
    jump_times = [0]

    # simulate for burn-in time and time between n samples
    while t < tmax + (n - 1) * ts:

        # current state
        x1, x2 = path[-1][0], path[-1][1]

        # transition rates
        q_tx_1 = params['k_tx_1']
        q_tx_2 = params['k_tx_2']
        q_deg_1 = x1 * params['k_deg_1']
        q_deg_2 = x2 * params['k_deg_2']
        q_reg = x1 * x2 * params['k_reg']
        q_hold = q_tx_1 + q_tx_2 + q_deg_1 + q_deg_2 + q_reg

        # holding time in current state
        t_hold = -np.log(rng.uniform()) / q_hold
        t += t_hold
        jump_times.append(t)

        # jump probability
        outcome = [1, 2, 3, 4, 5]
        prob = [
            q_tx_1 / q_hold,
            q_tx_2 / q_hold,
            q_deg_1 / q_hold,
            q_deg_2 / q_hold,
            q_reg / q_hold
        ]
        jump = rng.choice(outcome, p=prob)
        match jump:
            case 1:
                path.append((x1 + 1, x2))
            case 2:
                path.append((x1, x2 + 1))
            case 3:
                path.append((x1 - 1, x2))
            case 4:
                path.append((x1, x2 - 1))
            case 5:
                path.append((x1 - 1, x2 - 1))

    # take the transcript states
    x1_path = [state[0] for state in path]
    x2_path = [state[1] for state in path]

    # create step function of sample path from jump times and jump values
    x1_path_function = scipy.interpolate.interp1d(jump_times, x1_path, kind='previous')
    x2_path_function = scipy.interpolate.interp1d(jump_times, x2_path, kind='previous')

    # take values at sampling times as samples from stationary dist
    sample_times = [tmax + i * ts for i in range(n)]
    x1_samples = x1_path_function(sample_times)
    x2_samples = x2_path_function(sample_times)

    # convert to integers
    x1_samples = [int(x1) for x1 in x1_samples]
    x2_samples = [int(x2) for x2 in x2_samples]

    # re-combine to pairs of samples
    samples = list(zip(x1_samples, x2_samples))

    # plot sample paths
    if plot:
        x = np.linspace(0, tmax + (n - 1) * ts, 10000)
        plt.plot(x, x1_path_function(x), label="X1 sample path", color="blue")
        plt.plot(x, x2_path_function(x), label="X2 sample path", color="purple")
        #plt.axvline(tmax, label="Burn-in time", color="orange")
        plt.xlabel("Time")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

    return samples

# ------------------------------------------------
# Dataset simulation: interaction range
# ------------------------------------------------

def simulate_dataset_range_BD(interaction_values, cells=1000, rate=1, tqdm_disable=True):
    '''
    Produce a dataset of pairs of samples with fixed parameters (rate) over a
    range of interaction strength values.

    Args:
        cells: number of samples to simulate per gene-pair
        interaction_values: k_reg parameters values to simulate samples for
        rate: k_tx parameter values for all genes

    Returns:
        Dataset instance containing information as attributes

        params_df: pandas dataframe of model parameters per gene-pair
        counts_df: pandas dataframe of sampled counts per gene-pair
    '''

    # number of pairs
    gene_pairs = len(interaction_values)

    # dataframes
    params_df = pd.DataFrame(index=[f"Gene-pair-{i}" for i in range(gene_pairs)], columns=['k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg'])
    counts_df = pd.DataFrame(index=[f"Gene-pair-{i}" for i in range(gene_pairs)], columns=[f"Cell-{j}" for j in range(cells)])

    # for each gene
    for i in tqdm.tqdm(range(gene_pairs), disable=tqdm_disable):

        # Set reaction rate parameters
        k_tx_1 = rate
        k_tx_2 = rate
        k_deg_1 = 1
        k_deg_2 = 1
        k_reg = interaction_values[i]

        # store parameters
        params_df.iloc[i] = [k_tx_1, k_tx_2, k_deg_1, k_deg_2, k_reg]

        params = {
            'k_tx_1': k_tx_1,
            'k_tx_2': k_tx_2,
            'k_deg_1': k_deg_1,
            'k_deg_2': k_deg_2,
            'k_reg': k_reg
        }

        # simulate sample from model
        sample = gillespie_birth_death(params, cells)

        # store counts
        counts_df.iloc[i] = sample

    # construct dataset object
    data = dataset.Dataset()

    # store information
    data.count_dataset = counts_df
    data.param_dataset = params_df
    data.cells = cells
    data.gene_pairs = len(interaction_values)
    data.beta = np.array([1.0 for j in range(cells)])

    return data

def simulate_dataset_range_TE(interaction_values, cells=1000, gene=0.5, rate=1, tqdm_disable=True):
    '''
    Produce a dataset of pairs of samples with fixed parameters (rate) over a
    range of interaction strength values.

    Args:
        cells: number of samples to simulate per gene-pair
        interaction_values: k_reg parameters values to simulate samples for
        rate: k_tx parameter values for all genes

    Returns:
        Dataset instance containing information as attributes

        params_df: pandas dataframe of model parameters per gene-pair
        counts_df: pandas dataframe of sampled counts per gene-pair
    '''

    # number of pairs
    gene_pairs = len(interaction_values)

    # dataframes
    params_df = pd.DataFrame(index=[f"Gene-pair-{i}" for i in range(gene_pairs)], columns=['k_on_1', 'k_on_2', 'k_off_1', 'k_off_2', 'k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg'])
    counts_df = pd.DataFrame(index=[f"Gene-pair-{i}" for i in range(gene_pairs)], columns=[f"Cell-{j}" for j in range(cells)])

    # for each gene
    for i in tqdm.tqdm(range(gene_pairs), disable=tqdm_disable):

        # Set reaction rate parameters
        k_on_1 = gene
        k_on_2 = gene
        k_off_1 = 1 - gene
        k_off_2 = 1 - gene
        k_tx_1 = rate
        k_tx_2 = rate
        k_deg_1 = 1
        k_deg_2 = 1
        k_reg = interaction_values[i]

        # store parameters
        params_df.iloc[i] = [k_on_1, k_on_2, k_off_1, k_off_2, k_tx_1, k_tx_2, k_deg_1, k_deg_2, k_reg]

        params = {
            'k_on_1': k_on_1,
            'k_on_2': k_on_2,
            'k_off_1': k_off_1,
            'k_off_2': k_off_2,
            'k_tx_1': k_tx_1,
            'k_tx_2': k_tx_2,
            'k_deg_1': k_deg_1,
            'k_deg_2': k_deg_2,
            'k_reg': k_reg
        }

        # simulate sample from model
        sample = gillespie_telegraph(params, cells)

        # store counts
        counts_df.iloc[i] = sample

    # construct dataset object
    data = dataset.Dataset()

    # store information
    data.count_dataset = counts_df
    data.param_dataset = params_df
    data.cells = cells
    data.gene_pairs = len(interaction_values)
    data.beta = np.array([1.0 for j in range(cells)])

    return data

# ------------------------------------------------
# Dataset simulation: log-uniform parameters
# ------------------------------------------------

def simulate_dataset_sampled(gene_pairs=100, cells=1000, interaction_chance=0.5,
                             conditional=False, sig=0.5, scale=1, plot=False, tqdm_disable=True):
    '''
    Produce dataset of gene pairs' simulated parameters and samples.

    Produce a dataset of pairs of genes where for each pair parameters of a
    birth-death regulation model are simulated, and then a sample of size
    'cells' is simulated from the model. Parameters are sampled from log-uniform 
    distributions informed by single cell RNA sequencing experiments to produce
    realistic data. The paramters and samples are returned in a dictionary of 2
    pandas dataframes.

    Args:
        beta: per cell capture efficiency vector / single value
        gene_pairs: number of gene pairs to simulate
        cells: number of samples to simulate per gene pair
        interaction_chance: float in [0, 1], chance the interation parameter
                            'k_reg' is sampled vs being set to 0 (no interaction)
        conditional: toggle if model parameters for each gene in the pair are
                     sampled independently (False) or conditionally (True)
        sig: standard deviation about common value for parameters of each gene
             during conditional sampling 
        scale: scaling of transcription rate (previously mean capture)
        plot: toggle plotting a scatter plot of the simulated parameters

    Returns:
        Dataset instance containing information as attributes

        params_df: pandas dataframe of model parameters per gene-pair
        counts_df: pandas dataframe of sampled counts per gene-pair
    '''

    # initialize random generator
    rng = np.random.default_rng()

    # dataframes
    params_df = pd.DataFrame(index=[f"Gene-pair-{i}" for i in range(gene_pairs)], columns=['k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg'])
    counts_df = pd.DataFrame(index=[f"Gene-pair-{i}" for i in range(gene_pairs)], columns=[f"Cell-{j}" for j in range(cells)])

    # for each gene
    for i in tqdm.tqdm(range(gene_pairs), disable=tqdm_disable):

        # Select if interation or not
        u = rng.uniform()
        if u < interaction_chance:
            interaction = True
        else:
            interaction = False

        # Simulate reaction rate parameters 

        # conditional sampling
        if conditional:

            # sample log-mean capture efficiency
            log_mean = rng.uniform(-0.5, 1.5)

            # sample from normal distribution about the log-mean
            log_mean_1 = rng.normal(log_mean, sig)
            log_mean_2 = rng.normal(log_mean, sig)

            # sample degradation rates for each gene
            log_k_deg_1 = rng.uniform(-1, 0)
            log_k_deg_2 = rng.uniform(-1, 0)

            # compute transcription rates using log-mean and deg rate
            log_k_tx_beta_1 = log_mean_1 + log_k_deg_1
            log_k_tx_beta_2 = log_mean_2 + log_k_deg_2

            # sample interaction strength
            if interaction: log_k_reg = rng.uniform(-1, 1)

        # independent sampling
        else:

            # sample rates from log-uniform distribution for both genes
            log_k_tx_beta_1 = rng.uniform(-1, 1.5)
            log_k_tx_beta_2 = rng.uniform(-1, 1.5)
            log_k_deg_1 = rng.uniform(-1, 0)
            log_k_deg_2 = rng.uniform(-1, 0)
            if interaction: log_k_reg = rng.uniform(-2, 2)

        # exponentiate and scale
        k_tx_1 = (10 ** log_k_tx_beta_1) / scale
        k_tx_2 = (10 ** log_k_tx_beta_2) / scale
        k_deg_1 = 10 ** log_k_deg_1
        k_deg_2 = 10 ** log_k_deg_2
        if interaction:
            k_reg = 10 ** log_k_reg
        else:
            k_reg = 0

        # store parameters
        params_df.iloc[i] = [k_tx_1, k_tx_2, k_deg_1, k_deg_2, k_reg]

        params = {
            'k_tx_1': k_tx_1,
            'k_tx_2': k_tx_2,
            'k_deg_1': k_deg_1,
            'k_deg_2': k_deg_2,
            'k_reg': k_reg
        }

        # simulate sample from model
        samples = gillespie_birth_death(params, cells)

        # store counts
        counts_df.iloc[i] = samples

    # plot
    if plot:
        plt.scatter(np.log10(params_df['k_tx_1'].astype(np.float64)) - np.log10(params_df['k_deg_1'].astype(np.float64)), np.log10(params_df['k_tx_2'].astype(np.float64)) - np.log10(params_df['k_deg_2'].astype(np.float64)))
        plt.xlabel("log(k_tx_1 / k_deg_1)")
        plt.ylabel("log(k_tx_2 / k_deg_2)")
        plt.title("Scatter plot of gene-pair parameters")
        plt.show()

    # construct dataset object
    data = dataset.Dataset()

    # store information
    data.count_dataset = counts_df
    data.param_dataset = params_df
    data.cells = cells
    data.gene_pairs = gene_pairs
    data.beta = np.array([1.0 for j in range(cells)])
    
    return data
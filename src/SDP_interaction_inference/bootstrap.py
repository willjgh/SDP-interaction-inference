'''
Module to compute moment confidence intervals from data for use in optimization.

Given a sample of pairs of counts from a pair of genes, e.g. scRNA-seq data, 
use bootstrap resampling to compute confidence interval bounds on the moments
of the sample.

Typical example:

# get sample e.g. from dataset
sample = count_dataset.loc['Gene-pair-10']

# run bootstrap
bounds = bootstrap(sample)
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from interaction_inference import utils

# ------------------------------------------------
# Bootstrap moments
# ------------------------------------------------

def bootstrap(sample, d, resamples=None):
    '''
    Compute confidence intervals on the moments of a sample of count pairs, use
    resamples number of bootstrap resamples (default to sample size) and estimate
    moments up to order d.

    Args:
        sample: list of tuples (x1, x2) of integer counts per cell
        d: maximum moment order to estimate
        resamples: integer number of bootstrap resamples to use

    Returns:
        (2 x Nd) numpy array of CI bounds on each Nd moment of order <= d
    '''

    # get sample size
    n = len(sample)

    # get bootstrap size: default to sample size
    if resamples is None:
        resamples = n

    # helpful values
    powers = utils.compute_powers(S=2, d=d)
    Nd = utils.compute_Nd(S=2, d=d)

    # initialize random generator
    rng = np.random.default_rng()

    # convert string to tuple if neccessary (pandas reading csv to string)
    if type(sample[0]) == str:
        sample = [literal_eval(count_pair) for count_pair in sample]

    # separate sample pairs
    x1_sample = [x[0] for x in sample]
    x2_sample = [x[1] for x in sample]

    # convert sample to n x 2 array
    sample = np.array([x1_sample, x2_sample]).T

    # bootstrap to N x n x 2 array
    boot = rng.choice(sample, size=(resamples, n))

    # split into 2 N x n arrays
    x1_boot = boot[:, :, 0]
    x2_boot = boot[:, :, 1]

    # estimate
    moment_bounds = np.zeros((2, Nd))
    for i, alpha in enumerate(powers):

        # raise boot to powers
        x1_boot_alpha = x1_boot**alpha[0]
        x2_boot_alpha = x2_boot**alpha[1]

        # multiply (N x n)
        boot_alpha = x1_boot_alpha * x2_boot_alpha

        # mean over sample axis (N x 1)
        moment_estimates = np.mean(boot_alpha, axis=1)

        # quantile over boot axis (2 x 1)
        moment_interval = np.quantile(moment_estimates, [0.025, 0.975])

        # store
        moment_bounds[:, i] = moment_interval

    return moment_bounds
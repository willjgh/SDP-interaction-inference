'''
Module implementing class to handle correlation test inference method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import scipy
import tqdm
import pandas as pd
from ast import literal_eval

# ------------------------------------------------
# Correlation class
# ------------------------------------------------

class Correlation():
    def __init__(self, dataset, alternative="less", tqdm_disable=False, printing=True):
        '''Initialize analysis settings and result storage.'''

        # store reference to dataset
        self.dataset = dataset

        # analysis settings
        self.alternative = alternative

        # analyse dataset
        self.analyse_dataset(tqdm_disable=tqdm_disable, printing=printing)


    def analyse_dataset(self, tqdm_disable=False, printing=True):
        '''Analyse dataset using method settings and store results.'''

        # dict to store results
        solution_dict = {}

        # loop over dataset
        for i in tqdm.tqdm(range(self.dataset.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.dataset.count_dataset.loc[f'Gene-pair-{i}'])

            # convert string to tuple if neccessary (pandas reading csv to string)
            if type(sample[0]) == str:
                sample = [literal_eval(count_pair) for count_pair in sample]

            # separate pairs into individual samples
            x1_sample = [x[0] for x in sample]
            x2_sample = [x[1] for x in sample]

            # test
            pearson = scipy.stats.pearsonr(x1_sample, x2_sample, alternative=self.alternative)

            # store result
            solution_dict[i] = {'pvalue': float(pearson.pvalue), 'statistic': float(pearson.statistic)}

            if printing:
                print(f"sample {i} p-value: {pearson.pvalue}")

        # store results
        self.result_dict = solution_dict
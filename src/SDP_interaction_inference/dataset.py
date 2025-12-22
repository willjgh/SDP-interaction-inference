'''
Module implementing class to handle datasets and related settings.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_interaction_inference import bootstrap
import pandas as pd
import numpy as np
import tqdm

# ------------------------------------------------
# Dataset class
# ------------------------------------------------

class Dataset():
    def __init__(self):
        '''Initialise dataset settings'''

        # dataset iteself
        self.count_dataset = None
        self.param_dataset = None

        # size
        self.cells = None
        self.gene_pairs = None

        # capture efficiency
        self.beta = None

        # bootstrap settings
        self.resamples = None
        self.confidence = None
        self.d = None

        # moment bounds
        self.moment_bounds = {}

    def load_dataset(self, count_dataset_filename, beta=None, param_dataset_filename=None):
        '''Load dataset from csv files: paramter and count data'''
        self.count_dataset = pd.read_csv(count_dataset_filename, index_col=0)
        if param_dataset_filename:
            # parameter dataset only available for simulated data
            self.param_dataset = pd.read_csv(param_dataset_filename, index_col=0)

        # store shape and capture efficiency details
        self.gene_pairs, self.cells = self.count_dataset.shape
        self.beta = beta

    def store_dataset(self, count_dataset_filename, param_dataset_filename=None):
        '''Store dataset as csv files: parameter and count data'''
        self.count_dataset.to_csv(count_dataset_filename)
        if param_dataset_filename:
            # parameter dataset only available for simulated data
            self.param_dataset.to_csv(param_dataset_filename)

    def downsample(self, beta):
        '''
        Apply a beta capture efficiency to the dataset, returning a copy with
        binomially downsampled counts and corresponding beta stored.
        '''

        # fail if dataset already downsampled
        if not (self.beta == np.array([1.0 for j in range(self.cells)])).all():
            print("Dataset has already been downsampled")
            return None

        # fail if incomptible cell numbers
        if not (beta.shape[0] == self.cells):
            print("Incompatible cell numbers.")
            return None
        
        # initialize random generator
        rng = np.random.default_rng()

        # setup downsampled dataset
        downsampled_counts_df = pd.DataFrame(
            index=[f"Gene-pair-{i}" for i in range(self.gene_pairs)],
            columns=[f"Cell-{j}" for j in range(self.cells)]
        )

        # for each sample
        for i in range(self.gene_pairs):

            # extract counts
            sample = self.count_dataset.iloc[i]
            x1_sample = [x[0] for x in sample]
            x2_sample = [x[1] for x in sample]

            # downsample
            x1_sample_downsampled = rng.binomial(x1_sample, beta).tolist()
            x2_sample_downsampled = rng.binomial(x2_sample, beta).tolist()
            sample_downsampled = list(zip(x1_sample_downsampled, x2_sample_downsampled))
            
            # store counts
            downsampled_counts_df.iloc[i] = sample_downsampled

        # create new downsampled dataset object
        downsampled_dataset = Dataset()

        # store counts
        downsampled_dataset.count_dataset = downsampled_counts_df

        # store capture
        downsampled_dataset.beta = beta

        # copy over information
        downsampled_dataset.param_dataset = self.param_dataset
        downsampled_dataset.cells = self.cells
        downsampled_dataset.gene_pairs = self.gene_pairs

        return downsampled_dataset
    
    def bootstrap(self, d, tqdm_disable=True):
        '''
        For each sample in dataset compute bootstrap CI bounds on moments.

        Args:
            d: maximum moment order to estimate
        '''

        # store d
        self.d = d

        # loop over samples
        for i in tqdm.tqdm(range(self.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.count_dataset.loc[f'Gene-pair-{i}'])

            # bootstrap moments
            moments = bootstrap.bootstrap(
                sample,
                self.d,
                self.confidence,
                self.resamples
            )

            # store moments
            self.moment_bounds[f'sample-{i}'] = moments
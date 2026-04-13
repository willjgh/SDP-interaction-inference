'''
Module implementing class to handle MOSEK model free interacting optimization.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_interaction_inference import optimization_utils
from SDP_interaction_inference import utils
import tqdm
import numpy as np
import traceback
from time import time
from mosek.fusion import *
import mosek.fusion.pythonic
import scipy

# ------------------------------------------------
# Optimization class
# ------------------------------------------------

class MOSEKModelFreeInteracting():
    def __init__(
        self,
        dataset,
        d=None,
        d_bd=None,
        d_sd=None,
        method_opt="single",
        HAR=True,
        method_HAR="sphere",
        ACHR_warmup=10,
        N=1000,
        eps=10**-6,
        seed=None,
        time_limit=300,
        silent=True,
        printing=False,
        tqdm_disable=False
        ):
        '''Initialize analysis settings and result storage.'''
        
        # store reference to dataset
        self.dataset = dataset

        # moment order settings
        if d is not None:
            self.d = d
            self.d_bd = d
            self.d_sd = d
        elif (d_bd is not None) and (d_sd is not None):
            self.d = max(d_bd, d_me, d_sd)
            self.d_bd = d_bd
            self.d_sd = d_sd
        else:
            raise Exception(f"No moment order specified")

        # helpful values
        self.S = 2
        self.U = []
        self.Nd = utils.compute_Nd(self.S, self.d)
        self.Nbd = utils.compute_Nd(self.S, self.d_bd)
        self.B = optimization_utils.compute_B(self.dataset.beta, self.S, self.U, self.d_bd)

        # method settings
        self.N = N
        self.eps = eps
        self.HAR = HAR
        self.method_opt = method_opt
        self.method_HAR = method_HAR
        self.ACHR_warmup = ACHR_warmup
        self.time_limit = time_limit
        self.seed = seed

        # display settings
        self.silent = silent
        self.printing = printing
        self.tqdm_disable = tqdm_disable

        # results
        self.result_dict_optimization = {}
        self.result_dict_HAR          = {}

    def analyse_dataset(self):
        '''Analyse given dataset using method settings and store results.'''

        # loop over gene pairs in dataset
        for i in tqdm.tqdm(range(self.dataset.gene_pairs), disable=self.tqdm_disable):

            # optimize for feasible point(s) of sample i
            try:

                # find feasible points
                feasible_points = self.MOSEK_feasible_points(i)

                # compute feasible correlations
                feasible_correlations = [optimization_utils.MOSEK_compute_feasible_correlation(self.S, self.d, y) for y in feasible_points]

                # store
                self.result_dict_optimization[i] = {
                    'feasible_points': feasible_points,
                    'feasible_correlations': feasible_correlations
                }

                # Hit and Run from single feasible point
                if self.method_opt == "single" and self.HAR:

                    # hit and run
                    feasible_points_HAR, t_int_err, t_int_eps = self.hit_and_run(i, feasible_points[0])

                    # compute feasible correlations
                    feasible_correlations_HAR = [optimization_utils.MOSEK_compute_feasible_correlation(self.S, self.d, y) for y in feasible_points_HAR]

                    # store
                    self.result_dict_HAR[i] = {
                        'feasible_points': feasible_points_HAR,
                        'feasible_correlations': feasible_correlations_HAR,
                        'err': t_int_err,
                        'eps': t_int_eps
                    }

            # if exception
            except Exception as e:

                # display exception and traceback
                print(f"Optimization failed: {e}")
                traceback.print_exception(e)

                # store default result
                self.result_dict_optimization[i] = {
                    'feasible_points': None,
                    'feasible_correlations': None
                }
                self.result_dict_HAR[i] = {
                        'feasible_points': None,
                        'feasible_correlations': None,
                        'err': None,
                        'eps': None
                    }

    def MOSEK_feasible_points(self, i):

        # get moment bounds for sample i
        OB_bounds = self.dataset.moment_bounds[f'sample-{i}']

        # raise exception if moments not available
        if self.d_bd > self.dataset.d:
            raise Exception(f"Optimization uses bounds on d_bd = {self.d_bd} too high for dataset d = {self.dataset.d}")

        # store feasible points
        feasible_points = []

        with Model('MOSEK-SDP') as md: 

            # variables
            y = md.variable('y', self.Nd, Domain.greaterThan(0.))

            # variable dict
            variables = {
                'y': y
            }

            # moment matrices
            for s in range(self.S + 1):

                # restrict to d_sd
                M_s = optimization_utils.MOSEK_construct_M_s(y, s, self.S, self.d_sd)
                variables[f'M_{s}'] = M_s

                # PSD
                md.constraint(f'M_{s}_PSD', M_s == Domain.inPSDCone())

            # moment bounds

            # get CI bounds on OB moments (up to order d_bd)
            y_lb = OB_bounds[0, :self.Nbd]
            y_ub = OB_bounds[1, :self.Nbd]

            # moment bounds
            md.constraint('y_UB', self.B @ y[:self.Nbd] <= y_ub)
            md.constraint('y_LB', self.B @ y[:self.Nbd] >= y_lb)

            # fixed moment
            md.constraint('y0', y[0] == 1)

            # find feasible point(s)
            if self.method_opt == "single":

                # single feasible point
                md.objective(0)
                md.solve()
                feasible_points = [y.level()]

            elif self.method_opt == "index":

                # optimize min and max of each variable (index of y)
                for i in range(self.Nd):
                    
                    # minimize
                    md.objective(ObjectiveSense.Minimize, y[i])
                    md.solve()
                    feasible_points.append(y.level())

                    # maximize
                    md.objective(ObjectiveSense.Maximize, y[i])
                    md.solve()
                    feasible_points.append(y.level())

            elif self.method_opt == "random":

                # initialize random generator
                rng = np.random.default_rng(seed=self.seed)

                # optimize random linear objectives
                y_feas = []

                for i in range(self.N):
                    c = np.zeros(self.Nd)
                    c[:self.Nbd] = rng.uniform(-1, 1, size=self.Nbd)
                    md.objective(ObjectiveSense.Minimize, y.T @ c)
                    md.solve()
                    feasible_points.append(y.level())
            
            elif self.method_opt == "random_2":

                # optimize random linear objectives: only including order <2 moments
                y_feas = []

                # initialize random generator
                rng = np.random.default_rng(seed=self.seed)

                N2 = utils.compute_Nd(self.S, 2)
                for i in range(self.N):
                    c = np.zeros(self.Nd)
                    c[:N2] = rng.uniform(-1, 1, size=N2)
                    md.objective(ObjectiveSense.Minimize, y.T @ c)
                    md.solve()
                    feasible_points.append(y.level())

        return feasible_points

    def hit_and_run(self, i, y_feas):

        # initialize random generator
        rng = np.random.default_rng(seed=self.seed)

        # get moment bounds for sample i
        OB_bounds = self.dataset.moment_bounds[f'sample-{i}']

        # get constraint information
        y_lb = OB_bounds[0, :self.Nbd]
        y_ub = OB_bounds[1, :self.Nbd]

        # record feasible points
        feasible_points = [y_feas]

        # record numerical errors: err (tmin > 0 or tmax < 0), eps (|tmin - tmax| < eps)
        t_int_err = 0
        t_int_eps = 0

        # ACHR tracks centroid
        if self.method_HAR == "ACHR":
            center = y_feas

        # repeats
        for n in range(self.N - 1):

            # get current feasible point
            y_feas = feasible_points[-1]

            # sample random direction
            if self.method_HAR == "sphere" or (self.method_HAR == "ACHR" and n < self.warmup):

                # uniform on sphere
                v = rng.multivariate_normal(np.zeros(self.Nd), np.diag(np.ones(self.Nd)))
                v = v / np.linalg.norm(v)

            # ignore 1st dimension as fixed y[0] = 1
            v[0] = 0

            # linear t range: y_lb < B (y_0 + tv) < l_ub
            t_linear = np.zeros((self.Nd, 2))
            t_linear[0, :] = [-np.inf, np.inf]

            # compute constants
            l = y_lb - self.B @ y_feas
            u = y_ub - self.B @ y_feas
            Bv = self.B @ v

            # l < t Bv < u
            for i in range(1, self.Nd):
                if Bv[i] > 0:
                    t_linear[i, :] = [l[i] / Bv[i], u[i] / Bv[i]]
                else:
                    t_linear[i, :] = [u[i] / Bv[i], l[i] / Bv[i]]

            # semidefinite t range
            t_semi = np.zeros((self.S + 1, 2))

            # for each matrix
            for s in range(self.S + 1):

                # generalized eigenvalues
                C = optimization_utils.compute_M_s_value(y_feas, s, self.S, self.d)
                D = optimization_utils.compute_M_s_value(v, s, self.S, self.d)
                evals, _ = scipy.linalg.eig(C, -D)
                lam = np.array([np.real(ev) for ev in evals])

                # lam < 0
                lam_neg = lam[lam < 0]
                if lam_neg.size == 0:
                    t_semi[s, 0] = -np.inf
                else:
                    t_semi[s, 0] = np.max(lam_neg)
                
                # lam > 0
                lam_pos = lam[lam > 0]
                if lam_pos.size == 0:
                    t_semi[s, 1] = np.inf
                else:
                    t_semi[s, 1] = np.min(lam_pos)

            # intersect all t intervals
            t_min = max(np.max(t_linear[:, 0]), np.max(t_semi[:, 0]))
            t_max = min(np.min(t_linear[:, 1]), np.min(t_semi[:, 1]))

            # if width of t interval too small / numerical error: skip
            if (t_min > 0 or t_max < 0) and np.abs(t_max - t_min) < self.eps:
                t_int_err += 1
                t_int_eps += 1
                continue
            elif (t_min > 0 or t_max < 0):
                t_int_err += 1
                continue
            elif np.abs(t_max - t_min) < self.eps:
                t_int_eps += 1
                continue

            # uniformly sample feasible point along t line
            ts = rng.uniform(t_min, t_max)
            y_feas_new = y_feas + ts * v

            # store
            feasible_points.append(y_feas_new)

        return feasible_points, t_int_err, t_int_eps
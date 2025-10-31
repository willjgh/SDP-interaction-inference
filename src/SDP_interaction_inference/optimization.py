'''
Module implementing classes to handle optimization inference method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_interaction_inference import optimization_utils
from SDP_interaction_inference import utils
from SDP_interaction_inference.constraints import Constraint
import json
import tqdm
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import traceback
from time import time

# ------------------------------------------------
# Constants
# ------------------------------------------------

status_codes = {
    1: 'LOADED',
    2: 'OPTIMAL',
    3: 'INFEASIBLE',
    4: 'INF_OR_UNBD',
    5: 'UNBOUNDED',
    6: 'CUTOFF',
    7: 'ITERATION_LIMIT',
    8: 'NODE_LIMIT',
    9: 'TIME_LIMIT',
    10: 'SOLUTION_LIMIT',
    11: 'INTERRUPTED',
    12: 'NUMERIC',
    13: 'SUBOPTIMAL',
    14: 'INPROGRESS',
    15: 'USER_OBJ_LIMIT'
}

# ------------------------------------------------
# General Optimization class
# ------------------------------------------------

class Optimization():
    def __init__(
        self,
        dataset,
        constraints,
        reactions,
        vrs,
        db,
        R,
        S,
        U,
        d,
        fixed,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=100,
        silent=True,
        printing=False,
        tqdm_disable=False
        ):
        '''Initialize analysis settings and result storage.'''
        
        # store reference to dataset
        self.dataset = dataset

        # model settings
        self.constraints = constraints
        self.reactions = reactions
        self.vrs = vrs
        self.db = db
        self.R = R
        self.S = S
        self.U = U
        self.d = d
        self.fixed = fixed

        # optimization settings
        self.license_file = license_file
        self.time_limit = time_limit
        self.total_time_limit = total_time_limit
        self.eval_eps = eval_eps
        self.cut_limit = cut_limit
        self.K = K

        # display settings
        self.silent = silent
        self.printing = printing
        self.tqdm_disable = tqdm_disable

        # debugging
        self.eigenvalues = []

        # analyse dataset
        #self.analyse_dataset()


    def analyse_dataset(self):
        '''Analyse given dataset using method settings and store results.'''

        # dict to store results
        solution_dict = {}

        # loop over gene pairs in dataset
        for i in tqdm.tqdm(range(self.dataset.gene_pairs), disable=self.tqdm_disable):

            # test feasibility of sample i
            try:
                solution_dict[i] = self.feasibility_test(i)

            # if exception
            except Exception as e:

                # display exception and traceback
                print(f"Optimization failed: {e}")
                traceback.print_exception(e)

                # store default result
                solution_dict[i] = {
                    'status': None,
                    'time': None,
                    'cuts': None
                }

        # store as attribute
        self.result_dict = solution_dict


    def feasibility_test_old(self, i):
        '''
        Full feasibility test of birth death model via following algorithm

        Optimize NLP
        Infeasible: stop
        Feasible: check SDP feasibility
            Feasible: stop
            Infeasible: add cutting plane and return to NLP step

        Args:
            i: index of dataset sample to test
            
        Returns:
            dictionary of feasibility status and optimization time
        '''

        # get moment bounds for sample i
        OB_bounds = self.dataset.moment_bounds[f'sample-{i}']

        # raise exception if moments not available
        if self.d > self.dataset.d:
            raise Exception(f"Optimization d = {self.d} too high for dataset d = {self.dataset.d}")
        
        # adjust S = 2, U = [] bounds to optimization S, U (up to order d)
        OB_bounds = optimization_utils.bounds_adjust(OB_bounds, self.S, self.U, self.d)

        # if provided load WLS license credentials
        if self.license_file:
            environment_parameters = json.load(open(self.license_file))
        # otherwise use default environment (e.g Named User license)
        else:
            environment_parameters = {}
 
        # silence output
        if self.silent:
            environment_parameters['OutputFlag'] = 0

        # collect solution information
        solution = {
            'status': None,
            'time': None,
            'cuts': None
        }

        # environment context
        with gp.Env(params=environment_parameters) as env:

            # model context
            with gp.Model('test-SDP', env=env) as model:

                # construct base model (no semidefinite constraints)
                model, variables = optimization_utils.base_model(self, model, OB_bounds)
                
                # check feasibility
                model, status = optimization_utils.optimize(model)

                # collect solution information
                solution['status'] = status
                solution['time'] = model.Runtime

                # no semidefinite constraints: just return status
                if not self.constraints.moment_matrices:

                    if self.printing: print(status)

                    return solution

                # while feasible
                while status == "OPTIMAL":

                    if self.printing: print("NLP feasible")

                    # check semidefinite feasibility
                    model, semidefinite_feas = optimization_utils.semidefinite_cut(self, model, variables)

                    # semidefinite feasible
                    if semidefinite_feas:
                        break

                    # semidefinite infeasible
                    else:

                        # check feasibility with added cut
                        model, status = optimization_utils.optimize(model)

                        # update optimization time
                        solution['time'] += model.Runtime

                # if infeasible
                if status == "INFEASIBLE":
                    
                    if self.printing: print("SDP infeasible")

                    #model.computeIIS()
                    #model.write('test.ilp')

                # update final status
                solution['status'] = status

                # print
                if self.printing:
                    print(f"Optimization status: {solution['status']}")
                    print(f"Runtime: {solution['time']}")

                return status

    def feasibility_test(self, i):
        '''
        Full feasibility test of birth death model via following algorithm

        Optimize NLP
        Infeasible: stop
        Feasible: check SDP feasibility
            Feasible: stop
            Infeasible: add cutting plane and return to NLP step

        Args:
            i: index of dataset sample to test
            
        Returns:
            dictionary of feasibility status and optimization time
        '''

        # get moment bounds for sample i
        OB_bounds = self.dataset.moment_bounds[f'sample-{i}']

        # raise exception if moments not available
        if self.d > self.dataset.d:
            raise Exception(f"Optimization d = {self.d} too high for dataset d = {self.dataset.d}")
        
        # adjust S = 2, U = [] bounds to optimization S, U (up to order d)
        OB_bounds = optimization_utils.bounds_adjust(OB_bounds, self.S, self.U, self.d)

        # if provided load WLS license credentials
        if self.license_file:
            environment_parameters = json.load(open(self.license_file))
        # otherwise use default environment (e.g Named User license)
        else:
            environment_parameters = {}
 
        # silence output
        if self.silent:
            environment_parameters['OutputFlag'] = 0

        # environment context
        with gp.Env(params=environment_parameters) as env:

            # model context
            with gp.Model('test-SDP', env=env) as model:

                # construct base model (no semidefinite constraints)
                model, variables = optimization_utils.base_model(self, model, OB_bounds)
                
                # check feasibility
                model, status = optimization_utils.optimize(model)

                # collect solution information
                solution = {
                    'status': status,
                    'time': model.Runtime,
                    'cuts': 0
                }

                # no semidefinite constraints or non-optimal solution: return NLP status
                if not (self.constraints.moment_matrices and status == "OPTIMAL"):

                    return solution

                # while below time and cut limit
                while (solution['cuts'] < self.cut_limit) and (solution['time'] < self.total_time_limit):

                    # check semidefinite feasibility & add cuts if needed
                    model, semidefinite_feas = optimization_utils.semidefinite_cut(self, model, variables)

                    # cut
                    solution['cuts'] += 1

                    # semidefinite feasible: return
                    if semidefinite_feas:

                        return solution
                    
                    # semidefinite infeasible: check NLP feasibility with added cut
                    model, status = optimization_utils.optimize(model)

                    # update optimization time
                    solution['time'] += model.Runtime

                    # NLP + cut infeasible: return
                    # (also return for any other status, can only proceed if optimal as need feasible point)
                    if not (status == "OPTIMAL"):

                        # update solution
                        solution['status'] = status

                        return solution

                # set custom status
                if solution['cuts'] >= self.cut_limit:

                    # exceeded number of cutting plane iterations
                    solution['status'] = "CUT_LIMIT"
                
                elif solution['time'] >= self.total_time_limit:

                    # exceeded total optimization time
                    solution['status'] = "TOTAL_TIME_LIMIT"

                # print
                if self.printing:
                    print(f"Optimization status: {solution['status']}")
                    print(f"Runtime: {solution['time']}")

                return solution
            
# ------------------------------------------------
# Birth-Death Optimization subclass
# ------------------------------------------------

class BirthDeathOptimization(Optimization):

    def __init__(
        self,
        dataset,
        d,
        fixed=None,
        constraints=None,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=100,
        silent=True,
        printing=False,
        tqdm_disable=False
        ):

        # birth death settings
        reactions = [
            "1",
            "xs[0]",
            "1",
            "xs[1]",
            "xs[0] * xs[1]"
        ]
        vrs = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [-1, -1]
        ]
        db = 2
        R = 5
        S = 2
        U = []

        # default constraints if not modified
        if not constraints:
            constraints = Constraint(
                moment_bounds=True,
                moment_matrices=True,
                moment_equations=True
            )

        # default fixed values if not modified
        if not fixed:
            fixed = [(1, 1)]

        super().__init__(
            dataset,
            constraints,
            reactions,
            vrs,
            db,
            R,
            S,
            U,
            d,
            fixed,
            license_file,
            time_limit,
            total_time_limit,
            eval_eps,
            cut_limit,
            K,
            silent,
            printing,
            tqdm_disable
        )

# ------------------------------------------------
# Telegraph Optimization subclass
# ------------------------------------------------

class TelegraphOptimization(Optimization):

    def __init__(
        self,
        dataset,
        d,
        fixed=None,
        constraints=None,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=100,
        silent=True,
        printing=False,
        tqdm_disable=False
        ):

        # telegraph settings
        reactions = [
            "1 - xs[2]",
            "xs[2]",
            "xs[2]",
            "xs[0]",
            "1 - xs[3]",
            "xs[3]",
            "xs[3]",
            "xs[1]",
            "xs[0] * xs[1]"
        ]
        vrs = [
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [-1, -1, 0, 0]
        ]
        db = 2
        R = 9
        S = 4
        U = [2, 3]

        # default constraints if not modified
        if not constraints:
            constraints = Constraint(
                moment_bounds=True,
                moment_matrices=True,
                moment_equations=True,
                telegraph_moments=True
            )

        # default fixed values if not modified
        if not fixed:
            fixed = [(3, 1)]

        super().__init__(
            dataset,
            constraints,
            reactions,
            vrs,
            db,
            R,
            S,
            U,
            d,
            fixed,
            license_file,
            time_limit,
            total_time_limit,
            eval_eps,
            cut_limit,
            K,
            silent,
            printing,
            tqdm_disable
        )

# ------------------------------------------------
# Model-free Optimization subclass
# ------------------------------------------------

class ModelFreeOptimization(Optimization):

    def __init__(
        self,
        dataset,
        d,
        fixed=None,
        constraints=None,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=100,
        silent=True,
        printing=False,
        tqdm_disable=False
        ):

        # telegraph settings
        reactions = []
        vrs = []
        db = 0
        R = 0
        S = 2
        U = []

        # default constraints if not modified
        if not constraints:
            constraints = Constraint(
                moment_bounds=True,
                moment_matrices=True,
                factorization=True
            )

        # default fixed values if not modified
        if not fixed:
            fixed = []

        super().__init__(
            dataset,
            constraints,
            reactions,
            vrs,
            db,
            R,
            S,
            U,
            d,
            fixed,
            license_file,
            time_limit,
            total_time_limit,
            eval_eps,
            cut_limit,
            K,
            silent,
            printing,
            tqdm_disable
        )
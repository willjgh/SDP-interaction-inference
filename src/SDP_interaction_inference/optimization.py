'''
Module implementing classes to handle optimization inference method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_interaction_inference import optimization_utils
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
# Optimization class
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
        fixed=[],
        time_limit=300,
        eval_eps=10**-6,
        print_solution=False,
        license_file=None,
        silent=True,
        K=100,
        tqdm_disable=False,
        compute_IIS=False,
        write_model=False):
        '''Initialize analysis settings and result storage.'''
        
        # store reference to dataset
        self.dataset = dataset

        # constraint settings
        self.constraints = constraints

        # add storage of other attributes

        # analysis settings
        self.license_file = license_file
        self.time_limit = time_limit
        self.silent = silent
        self.K = K
        self.tqdm_disable = tqdm_disable
        self.print_solution = print_solution
        self.compute_IIS = compute_IIS
        self.write_model = write_model

        # analyse dataset
        self.analyse_dataset()


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
                    'time': 0.0
                }

        # store as attribute
        self.result_dict = solution_dict


    def feasibility_test(self, i):
        '''
        Full feasibility test of birth death model via following algorithm

        Optimize NLP
        Infeasible: stop
        Feasible: check SDP feasibility
            Feasible: stop
            Infeasible: add cutting plane and return to NLP step

        Args:
            OB_bounds: confidence intervals on observed moments up to order d (at least)
            beta: capture efficiency vector
            reactions: list of strings detailing a_r(x) for each reaction r
            vrs: list of lists detailing v_r for each reaction r
            db: largest order a_r(x)
            R: number of reactions
            S: number of species
            U: indices of unobserved species
            d: maximum moment order used
            fixed: list of pairs of (reaction index r, value to fix k_r to)
            time_limit: optimization time limit

            constraint options

            moment_bounds: CI bounds on moments
            moment_matrices: 
            moment_equations
            factorization
            factorization_telegraph
            telegraph_moments

            optimization options

            print_evals: toggle printing of moment matrix eigenvalues
            printing: toggle printing of feasibility status
            eval_eps: threshold of allowed negative eigenvalues for semidefinite
            
        Returns:
            dictionary of feasibility status and optimization time
        '''

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
            'time': None
        }

        # environment context
        with gp.Env(params=environment_parameters) as env:

            # model context
            with gp.Model('test-SDP', env=env) as model:

                # construct base model (no semidefinite constraints)
                model, variables = optimization_utils.base_model(
                     model,
                     self.constraints,
                     self.dataset.moment_bounds[f'sample-{i}'],
                     self.dataset.beta,
                     self.reactions,
                     self.vrs,
                     self.db,
                     self.R,
                     self.S,
                     self.U,
                     self.d,
                     self.fixed,
                     self.time_limit
                )
                
                # check feasibility
                model, status = optimization_utils.optimize(model)

                # no semidefinite constraints: just return status
                if not self.constraints['moment_matrices']:

                    if self.printing: print(status)

                    # collect solution information
                    solution['status'] = status
                    solution['time'] = model.Runtime

                    return solution

                # while feasible
                while status == "OPTIMAL":

                    if self.printing: print("NLP feasible")

                    # check semidefinite feasibility
                    model, semidefinite_feas = optimization_utils.semidefinite_cut(
                        model,
                        variables,
                        self.S,
                        self.print_evals,
                        self.eval_eps,
                        self.printing
                    )

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
                if self.print_solution:
                    print(f"Optimization status: {solution['status']}")
                    print(f"Runtime: {solution['time']}")

                return status
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
        fixed,
        d=None,
        d_bd=None,
        d_me=None,
        d_sd=None,
        fixed_correlation=None,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=100,
        function_inj=None,
        save_model=False,
        load_model=False,
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
        self.fixed = fixed
        self.fixed_correlation = fixed_correlation

        # moment order settings
        if d is not None:
            self.d = d
            self.d_bd = d
            self.d_me = d
            self.d_sd = d
        elif (d_bd is not None) and (d_me is not None) and (d_sd is not None):
            self.d = max(d_bd, d_me, d_sd)
            self.d_bd = d_bd
            self.d_me = d_me
            self.d_sd = d_sd
        else:
            raise Exception(f"No moment order specified")

        # optimization settings
        self.license_file = license_file
        self.time_limit = time_limit
        self.total_time_limit = total_time_limit
        self.eval_eps = eval_eps
        self.cut_limit = cut_limit
        self.K = K
        self.function_inj = function_inj

        # file settings
        self.save_model = save_model
        self.load_model = load_model

        # display settings
        self.silent = silent
        self.printing = printing
        self.tqdm_disable = tqdm_disable

        # results
        self.result_dict      = {}
        self.eigenvalues_dict = {}
        self.optim_times_dict = {}
        self.feasible_values_dict = {}

        # analyse dataset
        #self.analyse_dataset()


    def analyse_dataset(self):
        '''Analyse given dataset using method settings and store results.'''

        # loop over gene pairs in dataset
        for i in tqdm.tqdm(range(self.dataset.gene_pairs), disable=self.tqdm_disable):

            # test feasibility of sample i
            try:
                solution, eigenvalues, optim_times, feasible_values = self.feasibility_test(i)

                # compute final recovered correlation
                correlation = optimization_utils.compute_feasible_correlation(self, solution, feasible_values)
                solution['correlation'] = correlation

                # store
                self.result_dict[i]      = solution
                self.eigenvalues_dict[i] = eigenvalues
                self.optim_times_dict[i] = optim_times
                self.feasible_values_dict[i] = feasible_values

            # if exception
            except Exception as e:

                # display exception and traceback
                print(f"Optimization failed: {e}")
                traceback.print_exception(e)

                # store default result
                self.result_dict[i] = {
                    'status': None,
                    'time': None,
                    'cuts': None
                }
                self.eigenvalues_dict[i] = None
                self.optim_times_dict[i] = None

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

        # store information from SDP loop
        eigenvalues = []
        optim_times = []
        feasible_values = []

        # get moment bounds for sample i
        OB_bounds = self.dataset.moment_bounds[f'sample-{i}']

        # raise exception if moments not available
        if self.d_bd > self.dataset.d:
            raise Exception(f"Optimization uses bounds on d_bd = {self.d_bd} too high for dataset d = {self.dataset.d}")

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

                # if provided: load model
                if self.load_model:

                    # get model
                    model = gp.read(self.load_model, env)

                    # get variables
                    Nd = utils.compute_Nd(self.S, self.d)
                    variables = {
                        'y': gp.MVar([model.getVarByName(f'y[{i}]') for i in range(Nd)]),
                        'k': gp.MVar([model.getVarByName(f'k[{i}]') for i in range(self.R)])
                    }
                    if self.constraints.moment_matrices:
                        for s in range(self.S + 1):
                            M_s = optimization_utils.construct_M_s(variables['y'], s, self.S, self.d)
                            variables[f'M_{s}'] = M_s

                    # general setup
                    model.Params.TimeLimit = self.time_limit

                # otherwise: construct base model (no semidefinite constraints)
                else:
                    model, variables = optimization_utils.base_model(self, model, OB_bounds)

                # optional function injection to model
                if self.function_inj:
                    model, variables = self.function_inj(self, model, variables)
                
                # check feasibility
                model, status, var_dict = optimization_utils.optimize(model)

                # collect solution information
                solution = {
                    'status': status,
                    'time': model.Runtime,
                    'cuts': 0
                }

                optim_times.append(solution['time'])
                feasible_values.append(var_dict)

                # no semidefinite constraints or non-optimal solution: return NLP status
                if not (self.constraints.moment_matrices and status == "OPTIMAL"):

                    # save final model
                    if self.save_model:
                        model.write(self.save_model)

                    return solution, eigenvalues, optim_times, feasible_values

                # while below time and cut limit
                while (solution['cuts'] < self.cut_limit) and (solution['time'] < self.total_time_limit):

                    # check semidefinite feasibility & add cuts if needed
                    model, semidefinite_feas, evals_data = optimization_utils.semidefinite_cut(self, model, variables)

                    # store eigenvalue & optim time data
                    eigenvalues.append(evals_data)
                    optim_times.append(model.Runtime)

                    # semidefinite feasible: return
                    if semidefinite_feas:

                        # save final model
                        if self.save_model:
                            model.write(self.save_model)

                        return solution, eigenvalues, optim_times, feasible_values
                    
                    # record cut
                    solution['cuts'] += 1
                    
                    # semidefinite infeasible: check NLP feasibility with added cut
                    model, status, var_dict = optimization_utils.optimize(model)

                    # update optimization time
                    solution['time'] += model.Runtime

                    # store feasible values
                    feasible_values.append(var_dict)

                    # NLP + cut infeasible: return
                    # (also return for any other status, can only proceed if optimal as need feasible point)
                    if not (status == "OPTIMAL"):

                        # update solution
                        solution['status'] = status

                        # save final model
                        if self.save_model:
                            model.write(self.save_model)

                        return solution, eigenvalues, optim_times, feasible_values

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

                # save final model
                if self.save_model:
                    model.write(self.save_model)

                return solution, eigenvalues, optim_times, feasible_values
            
# ------------------------------------------------
# Birth-Death Optimization subclass
# ------------------------------------------------

class BirthDeathOptimization(Optimization):

    def __init__(
        self,
        dataset,
        d=None,
        d_bd=None,
        d_me=None,
        d_sd=None,
        fixed=None,
        fixed_correlation=None,
        constraints=None,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=100,
        function_inj=None,
        save_model=False,
        load_model=False,
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
            fixed,
            d,
            d_bd,
            d_me,
            d_sd,
            fixed_correlation,
            license_file,
            time_limit,
            total_time_limit,
            eval_eps,
            cut_limit,
            K,
            function_inj,
            save_model,
            load_model,
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
        d=None,
        d_bd=None,
        d_me=None,
        d_sd=None,
        fixed=None,
        fixed_correlation=None,
        constraints=None,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=100,
        function_inj=None,
        save_model=False,
        load_model=False,
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
            fixed,
            d,
            d_bd,
            d_me,
            d_sd,
            fixed_correlation,
            license_file,
            time_limit,
            total_time_limit,
            eval_eps,
            cut_limit,
            K,
            function_inj,
            save_model,
            load_model,
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
        d=None,
        d_bd=None,
        d_me=None,
        d_sd=None,
        fixed=None,
        fixed_correlation=None,
        constraints=None,
        license_file=None,
        time_limit=300,
        total_time_limit=300,
        eval_eps=10**-6,
        cut_limit=100,
        K=100,
        function_inj=None,
        save_model=False,
        load_model=False,
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
            fixed,
            d,
            d_bd,
            d_me,
            d_sd,
            fixed_correlation,
            license_file,
            time_limit,
            total_time_limit,
            eval_eps,
            cut_limit,
            K,
            function_inj,
            save_model,
            load_model,
            silent,
            printing,
            tqdm_disable
        )
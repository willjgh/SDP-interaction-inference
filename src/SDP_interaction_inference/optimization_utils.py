'''
Module to implement utility functions for optimization: constraints, etc.
'''
# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from SDP_interaction_inference import utils
import gurobipy as gp
from gurobipy import GRB
import sympy as sp
import numpy as np
import math
import time

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
# Constraint Functions
# ------------------------------------------------

def compute_A(alpha, reactions, vrs, db, R, S, d):
    '''
    Moment equation coefficient matrix
    NOTE: must have order of alpha <= d

    Args:
        alpha: moment order for equation (d/dt mu^alpha = 0)
        reactions: list of strings detailing a_r(x) for each reaction r
        vrs: list of lists detailing v_r for each reaction r
        db: largest order a_r(x)
        R: number of reactions
        S: number of species
        d: maximum moment order used (must be >= order(alpha) + db - 1)

    Returns:
        A: (R, Nd) matrix of coefficients
    '''

    if utils.compute_order(alpha) > d - db + 1:
        raise NotImplementedError(f"Maximum moment order {d} too small for moment equation of alpha = {alpha}: involves moments of higher order.")


    xs = sp.symbols([f'x{i}' for i in range(S)])

    # reaction propensity polynomials
    # props = [eval(str_ar) for str_ar in reactions]
    props = [sp.parse_expr(str_ar, {'xs': xs}) for str_ar in reactions]

    # number of moments of order <= d
    Nd = utils.compute_Nd(S, d)

    # get powers of order <= d
    powers = utils.compute_powers(S, d)

    # setup matrix
    A = np.zeros((R, Nd))

    for r, prop in enumerate(props):

        # expand b(x) * ((x + v_r)**alpha - x**alpha)
        term_1 = 1
        term_2 = 1
        for i in range(S):
            term_1 = term_1 * (xs[i] + vrs[r][i])**alpha[i]
            term_2 = term_2 * xs[i]**alpha[i]
        poly = sp.Poly(prop * (term_1 - term_2), xs)

        # loop over terms
        for xs_power, coeff in zip(poly.monoms(), poly.coeffs()):

            # get matrix index
            col = powers.index(list(xs_power))

            # store
            A[r, col] = coeff

    return A

def compute_B(beta, S, U, d):
    '''
    Capture efficiency moment scaling matrix

    Args:
        beta: per cell capture efficiency sample
        S: number of species
        U: unobserved species indices
        d: maximum moment order used

    Returns:
        B: (Nd, Nd) matrix of coefficients
    '''

    # number of moments of order <= d
    Nd = utils.compute_Nd(S, d)

    # compute powers of order <= d
    powers = utils.compute_powers(S, d)

    # compute beta moments of order <= d
    y_beta = np.zeros(d + 1)
    for l in range(d + 1):
        y_beta[l] = np.mean(beta**l)

    # setup matrix
    B = np.zeros((Nd, Nd))

    p = sp.Symbol('p')
    xs = sp.symbols([f'x{i}' for i in range(S)])

    # for each moment power
    for row, alpha in enumerate(powers):

        # setup polynomail
        poly_alpha = 1

        # for each species
        for i in range(S):

            # unobserved: no capture efficiency
            if i in U:
                moment = xs[i]**alpha[i]

            # observed: compute moment expression for E[Xi^alphai] in xi
            else:
                moment = utils.binomial_moment(xs[i], p, alpha[i])
            
            poly = sp.Poly(moment, p, xs[i])

            # multiply
            poly_alpha = poly_alpha * poly

        # loop over terms
        for (beta_power, *xs_power), coeff in zip(poly_alpha.monoms(), poly_alpha.coeffs()):

            # get matrix index
            col = powers.index(xs_power)

            B[row, col] += coeff * y_beta[beta_power]

    return B

def construct_M_s(y, s, S, d):
    '''Moment matrix variable constructor (s).'''
    if s == 0:
        D = math.floor(d / 2)
    else:
        D = math.floor((d - 1) / 2)
    powers_D = utils.compute_powers(S, D)
    powers_d = utils.compute_powers(S, d)
    ND = utils.compute_Nd(S, D)
    M_s = [[0 for j in range(ND)] for i in range(ND)]
    e_s = [1 if i == (s - 1) else 0 for i in range(S)]
    for alpha_index, alpha in enumerate(powers_D):
        for beta_index, beta in enumerate(powers_D):
            plus = utils.add_powers(alpha, beta, e_s, S=S)
            plus_index = powers_d.index(plus)
            M_s[alpha_index][beta_index] = y[plus_index].item()
    M_s = gp.MVar.fromlist(M_s)
    return M_s

# ------------------------------------------------
# Bounds adjustment
# ------------------------------------------------

def bounds_adjust(OB_bounds, S, U, d):
    '''
    Bootstrap gives (2, Nd) array of moments for S = 2, U = [].
    Adjust data into array of moment for different S and U
    '''

    # helpful values
    Nd = utils.compute_Nd(S, d)
    powers_S = utils.compute_powers(S, d)
    powers_2 = utils.compute_powers(2, d)

    # observed indices
    O = [i for i in range(S) if i not in U]

    # adjust bounds
    y_bounds = np.zeros((2, Nd))
    for i, alpha_S in enumerate(powers_S):

        # check if unobserved species present in moment
        unobserved_moment = False
        for j, alpha_j in enumerate(alpha_S):
            if (j in U) and (alpha_j > 0):
                unobserved_moment = True

        # unobserved: [0, inf] bounds
        if unobserved_moment:
            y_bounds[:, i] = np.array([0, np.inf])

        # otherwise: use data
        else:

            # get power for S = 2
            alpha_2 = [alpha_S[i] for i in O]
            j = powers_2.index(alpha_2)
            y_bounds[:, i] = OB_bounds[:, j]

    return y_bounds

# ------------------------------------------------
# General base model
# ------------------------------------------------

def base_model(opt, model, OB_bounds):
    '''
    Construct 'base model' with semidefinite constraints removed to give NLP

    Args:
        opt: Optimization class (or subclass), see relevant attributes
        model: empty gurobi model object
        OB_bounds: confidence intervals on observed moments up to order d (at least)

        Relevant class attributes

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

    Returns:
        model: gurobi model object with NLP constraints (all but semidefinite)
        variables: dict for model variable reference
    '''

    # model settings
    model.Params.TimeLimit = opt.time_limit

    # helpful values
    Nd = utils.compute_Nd(opt.S, opt.d)

    # variables
    y = model.addMVar(shape=Nd, vtype=GRB.CONTINUOUS, name="y", lb=0)
    k = model.addMVar(shape=opt.R, vtype=GRB.CONTINUOUS, name="k", lb=0, ub=opt.K)

    # variable dict
    variables = {
        'y': y,
        'k': k
    }

    if opt.constraints.moment_matrices:

        # moment matrices
        for s in range(opt.S + 1):
            M_s = construct_M_s(y, s, opt.S, opt.d)
            variables[f'M_{s}'] = M_s
    
    # constraints

    if opt.constraints.moment_bounds:

        '''
        # get CI bounds on OB moments (up to order d)
        y_lb = OB_bounds[0, :]
        y_ub = OB_bounds[1, :]

        # B scaling matrix
        B = compute_B(opt.dataset.beta, opt.S, opt.U, opt.d)

        # moment bounds
        model.addConstr(B @ y <= y_ub, name="y_UB")
        model.addConstr(B @ y >= y_lb, name="y_LB")
        '''

        # Alternate method:
        # do not adjust bounds (in optimization.py)
        # define downsampled moments y_d = B @ y
        # only explicitly bound observed, leave unobserved unbounded
        # avoids issues with e+100 upper bounds on unobserved moments
        # ------------------------------------------------------------

        # B scaling matrix
        B = compute_B(opt.dataset.beta, opt.S, opt.U, opt.d)

        # downsampled moments
        y_D = B @ y

        # bound
        O = [i for i in range(opt.S) if i not in opt.U]
        powers_S = utils.compute_powers(opt.S, opt.d)
        powers_2 = utils.compute_powers(2, opt.d)
        for i, alpha_S in enumerate(powers_S):
            # check if unobserved moment (non-zero power of unobserved species)
            observed = True
            for j, alpha_j in enumerate(alpha_S):
                if (j in opt.U) and (alpha_j > 0):
                    observed = False
            # observed: bound
            if observed:
                alpha_2 = [alpha_S[i] for i in O]
                j = powers_2.index(alpha_2)
                model.addConstr(y_D[i] <= OB_bounds[1, j], name=f"y_{i}_UB")
                model.addConstr(y_D[i] >= OB_bounds[0, j], name=f"y_{i}_LB")

        # -------------------------------------------------------------

    if opt.constraints.moment_equations:

        # moment equations (order(alpha) <= d - db + 1)
        moment_powers = utils.compute_powers(opt.S, opt.d - opt.db + 1)
        for alpha in moment_powers:
            A_alpha_d = compute_A(alpha, opt.reactions, opt.vrs, opt.db, opt.R, opt.S, opt.d)
            model.addConstr(k.T @ A_alpha_d @ y == 0, name=f"ME_{alpha}_{opt.d}")

    if opt.constraints.factorization:

        # factorization bounds
        powers = utils.compute_powers(opt.S, opt.d)
        for i, alpha in enumerate(powers):

            # E[X1^a1 X2^a2] = E[X1^a1] E[X2^a2]
            if (alpha[0] > 0) and (alpha[1] > 0):
                j = powers.index([alpha[0], 0])
                l = powers.index([0, alpha[1]])
                model.addConstr(y[i] == y[j] * y[l], name=f"Moment_factorization_{alpha[0]}_({alpha[1]})")

    if opt.constraints.telegraph_factorization:

        # factorization bounds
        powers = utils.compute_powers(opt.S, opt.d)
        for i, alpha in enumerate(powers):

            # E[X1^a1 X2^a2 G1^a3 G2^a4] = E[X1^a1 G1^a3] E[X2^a2 G2^a4]
            if (alpha[0] > 0) and (alpha[1] > 0):
                j = powers.index([alpha[0], 0, alpha[2], 0])
                l = powers.index([0, alpha[1], 0, alpha[3]])
                model.addConstr(y[i] == y[j] * y[l], name=f"Moment_factorization_{alpha[0], alpha[2]}_({alpha[1], alpha[3]})")

    if opt.constraints.telegraph_moments:

        # telegraph moment equality (as Gi in {0, 1}, E[Gi^n] = E[Gi] for n > 0, same with cross moments)
        powers = utils.compute_powers(opt.S, opt.d)
        for i, alpha in enumerate(powers):

            # G1, G2 powers > 0: equal to powers of 1
            if (alpha[2] > 0) and (alpha[3] > 0):
                j = powers.index([alpha[0], alpha[1], 1, 1])
                model.addConstr(y[i] == y[j], name="Telegraph_moment_equality_G1_G2")
            
            # G1 power > 0: equal to power of 1
            elif (alpha[2] > 0):
                j = powers.index([alpha[0], alpha[1], 1, alpha[3]])
                model.addConstr(y[i] == y[j], name="Telegraph_moment_equality_G1")

            # G2 power > 0: equal to power of 1
            elif (alpha[3] > 0):
                j = powers.index([alpha[0], alpha[1], alpha[2], 1])
                model.addConstr(y[i] == y[j], name="Telegraph_moment_equality_G2")

    if opt.constraints.telegraph_moments_ineq:

        # telegraph moment inequality (as Gi in {0, 1}, E[... Gi] <= E[...])
        powers = utils.compute_powers(opt.S, opt.d)
        for alpha_1 in range(opt.d + 1):
            for alpha_2 in range(opt.d - alpha_1 + 1):

                # E[... G1 G2] <= E[... G1]
                try:
                    i = powers.index([alpha_1, alpha_2, 1, 1])
                    j = powers.index([alpha_1, alpha_2, 1, 0])
                    model.addConstr(y[i] <= y[j], name="Telegraph_moment_inequality_G1G2_G1")
                except ValueError:
                    pass

                # E[... G1 G2] <= E[... G2]
                try:
                    i = powers.index([alpha_1, alpha_2, 1, 1])
                    r = powers.index([alpha_1, alpha_2, 0, 1])
                    model.addConstr(y[i] <= y[r], name="Telegraph_moment_inequality_G1G2_G2")
                except ValueError:
                    pass

                # E[... G1] <= E[...]
                try:
                    j = powers.index([alpha_1, alpha_2, 1, 0])
                    s = powers.index([alpha_1, alpha_2, 0, 0])
                    model.addConstr(y[j] <= y[s], name="Telegraph_moment_inequality_G1")
                except ValueError:
                    pass

                # E[... G2] <= E[...]
                try:
                    r = powers.index([alpha_1, alpha_2, 0, 1])
                    s = powers.index([alpha_1, alpha_2, 0, 0])
                    model.addConstr(y[r] <= y[s], name="Telegraph_moment_inequality_G2")
                except ValueError:
                    pass

    # fixed moment
    model.addConstr(y[0] == 1, name="y0_base")

    # fixed parameters
    for r, val in opt.fixed:
        model.addConstr(k[r] == val, name=f"k{r}_fixed")

    # fixed correlation
    if opt.fixed_correlation:

        # get variables
        powers = utils.compute_powers(opt.S, opt.d)
        if opt.S == 4:
            i_xy = powers.index([1, 1, 0, 0])
            i_x  = powers.index([1, 0, 0, 0])
            i_y  = powers.index([0, 1, 0, 0])
            i_x2 = powers.index([2, 0, 0, 0])
            i_y2 = powers.index([0, 2, 0, 0])
        elif opt.S == 2:
            i_xy = powers.index([1, 1])
            i_x  = powers.index([1, 0])
            i_y  = powers.index([0, 1])
            i_x2 = powers.index([2, 0])
            i_y2 = powers.index([0, 2])
        var_x = y[i_x2] - y[i_x]**2
        var_y = y[i_y2] - y[i_y]**2
        cov_xy = y[i_xy] - y[i_x] * y[i_y]

        # dummy zero variable:
        # GUROBI only supports non-linear expressions of the form:
        # variable = f(variables)
        # so 0 = f(variables) can only be done using a dummy zero variable
        z = model.addVar()
        model.addConstr(z == 0, name="Dummy_var")
        model.addConstr(z == opt.fixed_correlation**2 * var_x * var_y - cov_xy**2, name=f"Correlation_fixed"),
        if opt.fixed_correlation > 0:
            model.addConstr(cov_xy >= 0, name=f"Correlation_sign")
        else:
            model.addConstr(cov_xy <= 0, name=f"Correlation_sign")

    return model, variables

# ------------------------------------------------
# Optimization functions
# ------------------------------------------------

def optimize(model):
    '''Optimize model with no objective, return status & feasible point.'''

    # optimize
    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()
    status = status_codes[model.status]

    # get variable values
    all_vars = model.getVars()
    try:
        values = model.getAttr("X", all_vars)
    except:
        values = [None for var in all_vars]
    names = model.getAttr("VarName", all_vars)
    var_dict = {name: val for name, val in zip(names, values)}

    return model, status, var_dict

def semidefinite_cut(opt, model, variables):
    '''
    Check semidefinite feasibility of NLP feasible point
    Feasible: stop
    Infeasible: add cutting plane (ALL negative eigenvalues)

    Args:
        model: optimized NLP model
        variables: model variable reference dict
        print_evals: option to display moment matrix eigenvalues (semidefinite condition)

    Returns:
        model: model with any cutting planes added
        bool: semidefinite feasibility status
    '''

    # data list
    data = []

    # moment matrix values
    for s in range(opt.S + 1):
        data.append(
            {f'M_val': variables[f'M_{s}'].X}
        )

    # eigen information
    for s in range(opt.S + 1):
        evals_s, evecs_s = np.linalg.eigh(data[s]['M_val'])
        data[s]['evals'] = evals_s
        data[s]['evecs'] = evecs_s

    # extract eigenvalue data
    evals_data = {s: data[s]['evals'] for s in range(opt.S + 1)}

    if opt.printing:
        print("Moment matices eigenvalues:")
        for s in range(opt.S + 1):
            print(data[s]['evals'])

    # check if all positive eigenvalues
    positive = True
    for s in range(opt.S + 1):
        if not (data[s]['evals'] >= -opt.eval_eps).all():
            positive = False
            break

    # positive eigenvalues
    if positive:

        if opt.printing: print("SDP feasible\n")
    
        return model, True, evals_data

    # negative eigenvalue
    else:

        if opt.printing: print("SDP infeasible\n")

        # for each matrix
        for s in range(opt.S + 1):

            # for each M_s eigenvalue
            for i, lam in enumerate(data[s]['evals']):

                # if negative (sufficiently)
                if lam < -opt.eval_eps:

                    # get evector
                    v = data[s]['evecs'][:, i]

                    # add cutting plane
                    #model.addConstr(np.kron(v, v.T) @ variables[f'M_{s}'].reshape(-1) >= 0, name=f"Cut_{s}")
                    model.addConstr(v.T @ variables[f'M_{s}'] @ v >= 0, name=f"Cut_{s}")
                
                    if opt.printing: print(f"M_{s} cut added")

        if opt.printing: print("")

    return model, False, evals_data

# ------------------------------------------------
# Correlation computation
# ------------------------------------------------

def compute_feasible_correlation(opt, solution, feasible_values):
    '''Compute correlation value at feasible point.'''

    # only proceed if feasible point found
    if not (solution['status'] == "OPTIMAL"):
        return None
    
    # find indices of moments
    powers = utils.compute_powers(opt.S, opt.d)
    if opt.S == 4:
        i_xy = powers.index([1, 1, 0, 0])
        i_x  = powers.index([1, 0, 0, 0])
        i_y  = powers.index([0, 1, 0, 0])
        i_x2 = powers.index([2, 0, 0, 0])
        i_y2 = powers.index([0, 2, 0, 0])
    elif opt.S == 2:
        i_xy = powers.index([1, 1])
        i_x  = powers.index([1, 0])
        i_y  = powers.index([0, 1])
        i_x2 = powers.index([2, 0])
        i_y2 = powers.index([0, 2])

    # extract feasible point
    var_dict = feasible_values[-1]

    # collect moment values
    E_xy = var_dict[f'y[{i_xy}]']
    E_x  = var_dict[f'y[{i_x}]']
    E_y  = var_dict[f'y[{i_y}]']
    E_x2 = var_dict[f'y[{i_x2}]']
    E_y2 = var_dict[f'y[{i_y2}]']

    # compute statistics
    cov_xy = E_xy - E_x*E_y
    var_x = E_x2 - E_x**2
    var_y = E_y2 - E_y**2

    # return None if correlation undefined
    if var_x == 0 or var_y == 0:
        return None

    # compute correlation
    correlation = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))

    return correlation
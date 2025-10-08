'''
Module for dataclass recording the constraints of optimization models.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from dataclasses import dataclass, KW_ONLY

# ------------------------------------------------
# Constraint Names
# ------------------------------------------------

@dataclass
class Constraint:
    _: KW_ONLY

    # constraint options
    moment_bounds: bool = False
    moment_matrices: bool = False
    moment_equations: bool = False
    factorization: bool = False
    telegraph_factorization: bool = False
    telegraph_moments: bool = False
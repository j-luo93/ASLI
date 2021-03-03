from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from typing import ClassVar, List, Set, Dict, Optional, Union

from ortools.linear_solver import pywraplp

from dev_misc import add_argument, g
from sound_law.data.alphabet import Alphabet
from sound_law.main import setup
from sound_law.rl.action import SoundChangeAction, SoundChangeActionSpace
from sound_law.rl.mcts_cpp import \
    PyNull_abc  # pylint: disable=no-name-in-module
from sound_law.rl.trajectory import VocabState
from sound_law.train.manager import OnePairManager

def match_rulesets(gold: List[List[Action]], cand: List[Action], env: SoundChangeEnv) -> List[Tuple[Int, Tuple[Int]]]:
    '''Finds the optimal matching of rule blocks in the gold ruleset to 0, 1, or 2 rules in the candidate ruleset. Frames the problem as an integer linear program. Returns a list of tuples with the matching.'''
    solver = pywraplp.Solver.CreateSolver('SCIP') # FIXME investigate other solvers
    # form the different variables in this ILP
    # this dict maps strings to pointers to the variable that string represents. Makes things much more readable.
    v = {}
    # this dict maps rules/blocks to their individual constraint: eg 'gold_0' for the constraint for the 0th gold block, or 'cand_3' for the 3th candidate rule.
    c = {}
    # initialize constraints for all blocks/rules
    # constraint is of form a_i0 + ... + a_im + b_i(01) + ... <= 1
    # or of form a_0i + ... + a_ni + b_0(0i) + ... + b_0(in) <= 1
    # one such constraint exists for each gold block/cand rule. Only one of the variables a/b can be equal to 1, so only one matching occurs, if any. 
    for i in range(gold):
        c['gold_' + i] = solver.Constraint(0, 1) # FIXME investigate if the range (-solver.infinity(), 1) leads to better performance: all the examples use neg infty as the lower bound, even though here <0 isn't attainable.
    for j in range(cand):
        c['cand_' + j] = solver.Constraint(0, 1)

    # FIXME work on this after actually merging and determining what the SoundChangeEnv provides
    curr_state = env.init_state

    # create variables
    # FIXME also calculate distances while iterating to put into the objective function?
    for i, block in enumerate(gold):
        gold_state = None # FIXME
        for j, rule in enumerate(cand):
            a_var = 'a_' + i + j
            v[a_var] = solver.IntVar(0, 1, a_var)
            c['gold_' + i].SetCoefficient(v[a_var], 1)
            c['cand_' + j].SetCoefficient(v[a_var], 1)
        for j, rule1 in enumerate(cand):
            for k, rule2 in enumerate(cand[j+1:]):
                b_var = 'b_' + i + '(' + j + k + ')'
                v[b_var] = solver.IntVar(0, 1, b_var)
                c['gold_' + i].SetCoefficient(v[b_var], 1)
                c['cand_' + j].SetCoefficient(v[b_var], 1)
                c['cand_' + k].SetCoefficient(v[b_var], 1)

    # calculate the distances to create the objective function

    # solve the ILP

    objective = solver.Objective()

    # the solver can only maximize, so we input the coefficients as the negative distances so it's effectively a minimiziation problem
    objective.SetCoefficient(var, dist) # do this for all variables 

    objective.SetMaximization()
    solver.Solve()

    # reconstruct the solution and return it
    print('Maximum objective function value = %d' % solver.Objective().Value())
    print()
    # Print the value of each variable in the solution.
    for variable in v.keys():
        print('%s = %d' % (variable.name(), variable.solution_value()))
    # [END print_solution]
    
    return

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from typing import ClassVar, List, Set, Dict, Optional, Union

from ortools.linear_solver import pywraplp
import random

# from dev_misc import add_argument, g
# from sound_law.data.alphabet import Alphabet
# from sound_law.main import setup
# from sound_law.rl.action import SoundChangeAction, SoundChangeActionSpace
# from sound_law.rl.env import ToyEnv
# from sound_law.rl.mcts_cpp import \
#     PyNull_abc  # pylint: disable=no-name-in-module
# from sound_law.rl.trajectory import VocabState
# from sound_law.train.manager import OnePairManager


class ToyEnv():

    def __init__(self, init_state):
        self.init_state = init_state

    def apply_action(self, act, state):
        # somehow apply action to state
        new_state = state
        return new_state
    
    def apply_block(self, block, state):
        '''Applies a block of actions in order'''
        for act in block:
            state = self.apply_action(act, state)
        return state

    def dist_between(self, state1, state2):
        # somehow compute the edit distance between these two states
        return random.random() * random.randint(1, 20)

    def compare_effects(self, act1, act2, state):
        state1 = self.apply_action(act1, state)
        state2 = self.apply_action(act2, state)
        return self.dist_between(state1, state2)


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
    for i in range(len(gold)):
        c['gold_' + str(i)] = solver.Constraint(1, 1) # FIXME investigate if the range (-solver.infinity(), 1) leads to better performance: all the examples use neg infty as the lower bound, even though here <0 isn't attainable.
    for j in range(len(cand)):
        c['cand_' + str(j)] = solver.Constraint(0, 1)

    # TODO implement real SoundChangeEnv; currently using toy data "ToyEnv"
    curr_state = env.init_state
    objective = solver.Objective()

    # create variables
    # FIXME also calculate distances while iterating to put into the objective function?
    for i, block in enumerate(gold):
        gold_state = env.apply_block(block, curr_state)
        for j, rule in enumerate(cand):
            a_var = 'a_' + str(i) + str(j)
            v[a_var] = solver.IntVar(0, 1, a_var)
            c['gold_' + str(i)].SetCoefficient(v[a_var], 1)
            c['cand_' + str(j)].SetCoefficient(v[a_var], 1)
            # calculate edit distance and add to Objective function
            cand_state = env.apply_action(rule, curr_state)
            dist = env.dist_between(gold_state, cand_state)
            objective.SetCoefficient(v[a_var], dist)

        for j, rule1 in enumerate(cand):
            for k, rule2 in enumerate(cand[j+1:], start=j+1):
                b_var = 'b_' + str(i) + '(' + str(j) + ',' + str(k) + ')'
                v[b_var] = solver.IntVar(0, 1, b_var)
                c['gold_' + str(i)].SetCoefficient(v[b_var], 1)
                c['cand_' + str(j)].SetCoefficient(v[b_var], 1)
                c['cand_' + str(k)].SetCoefficient(v[b_var], 1)

                cand_state = env.apply_block([rule1, rule2], curr_state)
                dist = env.dist_between(gold_state, cand_state)
                objective.SetCoefficient(v[b_var], dist)
        # update state and continue
        curr_state = gold_state

    # solve the ILP
    objective.SetMinimization()
    solver.Solve()

    # reconstruct the solution and return it
    print('Minimum objective function value = %d' % solver.Objective().Value())
    # print()
    # # Print the value of each variable in the solution.
    # for variable in v.keys():
    #     print('%s = %d' % (variable.name(), variable.solution_value()))
    # # [END print_solution]

    # interpret solution as a matching:
    # TODO implement, for now just print out
    for name, var in v.items():
        if var.solution_value():
            print('%s = %d' % (var.name(), var.solution_value()))
    
    return

# toy data
# gold = [
#     ['a', 'b'],
#     ['c', 'd'],
#     ['e', 'f']
# ]
# cand = ['alpha', 'beta', 'gamma', 'delta']
gold = [[x,x] for x in range(100)]
cand = [x for x in range(100)]
env = ToyEnv('foo')

match_rulesets(gold, cand, env)

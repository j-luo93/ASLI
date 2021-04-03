from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from typing import ClassVar, List, Set, Dict, Optional, Union
import pandas as pd

from ortools.linear_solver import pywraplp
import random
import bisect

from dev_misc import add_argument, g
from sound_law.main import setup
# from sound_law.data.alphabet import Alphabet
# from sound_law.rl.action import SoundChangeAction, SoundChangeActionSpace
# from sound_law.rl.env import ToyEnv
# from sound_law.rl.mcts_cpp import \
#     PyNull_abc  # pylint: disable=no-name-in-module
# from sound_law.rl.trajectory import VocabState
import sound_law.rl.rule as rule

class ToyEnv():

    def __init__(self, start_state):
        self.start = start_state

    def apply_action(self, state, act):
        # somehow apply action to state
        new_state = state
        return new_state
    
    def apply_block(self, state, block):
        '''Applies a block of actions in order'''
        for act in block:
            state = self.apply_action(state, act)
        return state

    def get_state_edit_dist(self, state1, state2):
        # somehow compute the edit distance between these two states
        return (random.random() + 1) * random.randint(1, 20)

    # def compare_effects(self, act1, act2, state):
    #     state1 = self.apply_action(state, act1)
    #     state2 = self.apply_action(state, act2)
    #     return self.get_state_edit_dist(state1, state2)


def match_rulesets(gold: List[List[SoundChangeAction]],
                   cand: List[SoundChangeAction], 
                   env: SoundChangeEnv,
                   match_proportion: float = .7,
                   k_matches: int = 10) -> List[Tuple[Int, Tuple[Int]]]:
    '''Finds the optimal matching of rule blocks in the gold ruleset to 0, 1, or 2 rules in the candidate ruleset. Frames the problem as an integer linear program. Returns a list of tuples with the matching.'''

    solver = pywraplp.Solver.CreateSolver('SCIP') # TODO investigate other solvers
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
        c['gold_' + str(i)] = solver.Constraint(0, 1)
    for j in range(len(cand)):
        c['cand_' + str(j)] = solver.Constraint(0, 1)
    # finally, this matching constraint forces the model to match at least some of the rules (otherwise it would just match no rules to vacuously achieve a minimum objective of 0)
    # it stipulates that the sum of all variables must be >= some minimum match number
    # by the handshake lemma, only a constraint needs to be placed on gold — this implies some amount of matching with candidate
    min_match_number = int(match_proportion * len(gold))
    c['min_match'] = solver.Constraint(min_match_number, len(gold))

    # TODO implement real SoundChangeEnv; currently using toy data "ToyEnv"
    curr_state = env.start
    objective = solver.Objective()

    for i in range(len(gold)):
        # as an optimization, we only create variables for the best k_matches that a given gold block has with collections of rules in candidate. We assume that matchings with higher cost would never be chosen anyway and won't affect the solution, so they can just be excluded from the linear program.
        highest_cost = None
        paired_costs = [] # entries are of form (varname, i, [j...k], cost) — ie the variable pairing i with rules [j...k] has cost coefficient cost. Costs are in increasing order.

        block = gold[i]
        gold_state = env.apply_block(curr_state, block)
        for j in range(len(cand)):
            rule = cand[j]
            a_var_name = 'a_' + str(i) + ',' + str(j)
            cand_state = env.apply_action(curr_state, rule)
            cost = env.get_state_edit_dist(gold_state, cand_state)
            new_tuple = (a_var_name, i, [j], cost)

            # add this cost to the list if it's better than what we currently have
            if len(paired_costs) < k_matches or cost < highest_cost:
                if len(paired_costs) == k_matches:
                    del paired_costs[-1]
                bisect.insort_left(paired_costs, new_tuple) # insert in sorted order
                highest_cost = paired_costs[-1][3] # update costs
        
        for j in range(len(cand)):
            rule1 = cand[j]
            for k in range(j+1, len(cand)):
                rule2 = cand[k]
                b_var_name = 'b_' + str(i) + ',(' + str(j) + ',' + str(k) + ')'
                cand_state = env.apply_block(curr_state, [rule1, rule2])
                cost = env.get_state_edit_dist(gold_state, cand_state)
                new_tuple = (b_var_name, i, [j,k], cost)

                if len(paired_costs) < k_matches or cost < highest_cost:
                    if len(paired_costs) == k_matches:
                        del paired_costs[-1]
                    bisect.insort_left(paired_costs, new_tuple)
                    highest_cost = paired_costs[-1][3]
        
        for j in range(len(cand)):
            rule1 = cand[j]
            for k in range(j+1, len(cand)):
                rule2 = cand[k]
                for l in range(k+1, len(cand)):
                    rule3 = cand[l]
                    c_var_name = 'c_' + str(i) + ',(' + str(j) + ',' + str(k) + ',' + str(l) + ')'
                    cand_state = env.apply_block(curr_state, [rule1, rule2, rule3])
                    cost = env.get_state_edit_dist(gold_state, cand_state)
                    new_tuple = (c_var_name, i, [j,k,l], cost)

                    if len(paired_costs) < k_matches or cost < highest_cost:
                        if len(paired_costs) == k_matches:
                            del paired_costs[-1]
                        bisect.insort_left(paired_costs, new_tuple)
                        highest_cost = paired_costs[-1][3]
        
        # now that we have the k matchings with the lowest edit distance with this particular gold block, we can add the variables corresponding to these matchings to each of the relevant constraints:
        for var_name, i, cand_rules, cost in paired_costs:
            v[var_name] = solver.IntVar(0, 1, var_name)
            c['gold_' + str(i)].SetCoefficient(v[var_name], 1)
            for rule_index in cand_rules:
                c['cand_' + str(rule_index)].SetCoefficient(v[var_name], 1)
            c['min_match'].SetCoefficient(v[var_name], 1)
            objective.SetCoefficient(v[var_name], cost)

        # update the state and continue onto the next block in gold
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

    # interpret solution as a matching, returning a list pairing indices of blocks in gold to a list of indices of matched rules in cand
    # TODO(djwyen) implement this, for now just print out the soln
    for name, var in v.items():
        if var.solution_value():
            print('%s = %d' % (var.name(), var.solution_value()))
    
    return

if __name__ == "__main__":

    manager, gold, states, refs = rule.simulate()
    initial_state = states[0]

    gold = [[x,x] for x in range(10)]
    cand = [x for x in range(20)]
    env = ToyEnv('foo')

    match_rulesets(gold, cand, env)

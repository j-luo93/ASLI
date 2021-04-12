from __future__ import annotations

import bisect
import logging
import pickle
import random
import re
import typing
from collections import Counter
from dataclasses import astuple, dataclass, field
from itertools import chain, combinations
from typing import ClassVar, Dict, Iterator, List, Optional, Set, Tuple, Union

import pandas as pd
import sound_law.rl.rule as rule
from dev_misc import add_argument, g
from ortools.linear_solver import pywraplp
from sound_law.main import setup
# from sound_law.data.alphabet import Alphabet
from sound_law.rl.action import SoundChangeAction
from sound_law.rl.env import SoundChangeEnv
from sound_law.rl.rule import HandwrittenRule


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


def read_rules_from_txt(filename: str) -> List[SoundChangeAction]:
    '''Reads rules from a given file. Currently assuming file is a list of rules with commas at the end, formatted the same way test_annotations.csv is with [ruletype]: a > b / [context] _ [context], eg basic: z > ∅ / [+syllabic] r _ # '''
    rules = []
    with open(filename, 'r') as f:
        for line in f:
            rules.append(HandwrittenRule.from_str(line).to_action())
    return rules


@dataclass(order=True)
class MatchCandidate:
    cost: float
    var_name: str
    gold_index: int
    cand_indices: List[int]


def match_rulesets(gold: List[List[SoundChangeAction]],
                   cand: List[SoundChangeAction],
                   env: SoundChangeEnv,
                   match_proportion: float = .7,
                   k_matches: int = 10,
                   max_power_set_size: int = 3,
                   use_greedy_growth: bool = False,
                   interpret_matching: bool = False,
                   silent: bool = False,
                   null_only: bool = False) -> Tuple[List[Tuple[int, List[int]]], int, float, float, Dict[int, int]]:
    '''Finds the optimal matching of rule blocks in the gold ruleset to 0, 1, or 2 rules in the candidate ruleset. Frames the problem as an integer linear program. Returns a list of tuples with the matching.'''

    solver = pywraplp.Solver.CreateSolver('SCIP')  # TODO investigate other solvers
    # form the different variables in this ILP
    # this dict maps strings to pointers to the variable that string represents. Makes things much more readable.
    v = {}
    # this dict maps rules/blocks to their individual constraint: eg 'gold_0' for the constraint for the 0th gold block, or 'cand_3' for the 3th candidate rule.
    c = {}

    # initialize constraints for all blocks/rules
    # constraint is of form a_i0 + ... + a_im + b_i(01) + ... <= 1
    # cand constraint is of form a_0i + ... + a_ni + b_0(0i) + ... + b_0(in) <= 1
    # one such constraint exists for each gold block/cand rule. Only one of the variables a/b can be equal to 1, so only one matching occurs, if any
    for i in range(len(gold)):
        c['gold_' + str(i)] = solver.Constraint(0, 1)
    for j in range(len(cand)):
        c['cand_' + str(j)] = solver.Constraint(0, 1)

    # finally, this matching constraint forces the model to match at least some of the rules (otherwise it would just match no rules to vacuously achieve a minimum objective of 0)
    # it stipulates that the sum of all variables must be >= some minimum match number
    # by the handshake lemma, only a constraint needs to be placed on gold — this implies some amount of matching with candidate
    # we will update the actual bounds later based on how many gold blocks are actually active in this vocab, i.e. how many gold blocks actually apply to one or more words in the vocab
    c['min_match'] = solver.Constraint(0, 1)
    number_active_gold_blocks = 0  # counts the number of gold blocks eligible for matching, ie rules that actually change words

    curr_state = env.start
    objective = solver.Objective()
    size_cnt = Counter()
    total_null_costs = 0.0
    for i in range(len(gold)):
        # as an optimization, we only create variables for the best k_matches that a given gold block has with collections of rules in candidate. We assume that matchings with higher cost would never be chosen anyway and won't affect the solution, so they can just be excluded from the linear program.
        highest_cost = None
        # entries are of form (varname, i, [j...k], cost) — ie the variable pairing i with rules [j...k] has cost coefficient cost. Costs are in increasing order.
        paired_costs = []
        block = gold[i]
        # print('block', i, block)
        try:
            gold_state = env.apply_block(curr_state, block)
            null_cost = env.get_state_edit_dist(gold_state, curr_state)
            total_null_costs += null_cost
        except RuntimeError:
            # this block in gold doesn't actually apply to any items, causing the RuntimeError
            # it's nonsensical to discuss what rules are most similar to a block that does nothing so we skip this block: we don't match it with anything, and we don't even give it a variable for the ILP
            pass
        else:
            number_active_gold_blocks += 1
            # Match gold with a null candidate rule. Note that there is no constraint on the candidate side.
            var_name = f'pss0_{i},(-1)'  # -1 stands for the null candidate rule.
            v[var_name] = null_match = solver.IntVar(0, 1, var_name)
            c[f'gold_{i}'].SetCoefficient(null_match, 1)
            c['min_match'].SetCoefficient(null_match, null_cost)
            objective.SetCoefficient(null_match, null_cost)

            if null_only:
                # update the state and continue onto the next block in gold
                curr_state = gold_state
                continue

            # actually loop over the variables and create variables for this block

            def generate_match_candidates(var_name_prefix: str, power_set_size: int,
                                          bases: Optional[List[MatchCandidate]] = None) -> Iterator[MatchCandidate]:
                """Generate new match candidates. If `bases` s provided, grow them by adding one more rule (`power_set_size` is ignored in this case)."""
                # Each item in the iterator is a list (of size `power_set_size` if `bases` is None, otherwise one more than `bases`)
                # of (index, action) tuples, where `index` is the action's index in `cand`.
                iterator: Iterator[List[Tuple[int, SoundChangeAction]]]
                if bases is None:
                    iterator = combinations(enumerate(cand), power_set_size)
                else:
                    def grow_from_base(base: MatchCandidate) -> Iterator[List[Tuple[int, SoundChangeAction]]]:
                        last_j = base.cand_indices[-1]
                        base_block = [(ind, cand[ind]) for ind in base.cand_indices]
                        for j, rule in enumerate(cand[last_j + 1:], last_j + 1):
                            yield base_block + [(j, rule)]

                    iterator = chain(*[grow_from_base(base) for base in bases])

                for combo in iterator:
                    cand_indices, cand_rules = list(zip(*combo))
                    try:
                        cand_state = env.apply_block(curr_state, cand_rules)
                    except RuntimeError:
                        # this rule doesn't change anything, ie it has zero application sites. That causes the RuntimeError to be thrown.
                        # exclude this rule from consideration since it shouldn't be matched
                        pass
                    else:
                        cost = env.get_state_edit_dist(gold_state, cand_state)
                        var_name = f'{var_name_prefix}_{i},({",".join(map(str, cand_indices))})'
                        yield MatchCandidate(cost, var_name, i, cand_indices)

            def update_top_candidates(highest_cost: float, new_candidates: List[MatchCandidate]) -> float:
                """Update top match candidates. Remember to return the updated `highest_cost` to the caller."""
                for new_tuple in new_candidates:
                    cost = new_tuple.cost
                    # add this cost to the list if it's better than what we currently have
                    if len(paired_costs) < k_matches or cost < highest_cost:
                        if len(paired_costs) == k_matches:
                            del paired_costs[-1]
                        bisect.insort_left(paired_costs, new_tuple)  # insert in sorted order
                        highest_cost = paired_costs[-1].cost  # update costs
                return highest_cost

            if use_greedy_growth:
                power_set_size = 1
                pss1_seeds = generate_match_candidates('pss1', power_set_size)
                highest_cost = update_top_candidates(highest_cost, pss1_seeds)
                seeds = paired_costs
                while seeds and power_set_size < max_power_set_size:
                    power_set_size += 1
                    highest_cost = update_top_candidates(highest_cost,
                                                         generate_match_candidates(f'pss{power_set_size}', power_set_size, bases=seeds))
                    seeds = [match for match in paired_costs if len(match.cand_indices) == power_set_size]
            else:
                # This stores all pairs of `var_name_prefix` and `power_set_size`. "pss" stands for power set size.
                config_pairs = [(f'pss{l}', l) for l in range(1, max_power_set_size + 1)]
                for var_name_prefix, power_set_size in config_pairs:
                    highest_cost = update_top_candidates(highest_cost,
                                                         generate_match_candidates(var_name_prefix, power_set_size))
            # now that we have the k matchings with the lowest edit distance with this particular gold block, we can add the variables corresponding to these matchings to each of the relevant constraints:
            for match_cand in paired_costs:
                cost, var_name, i, cand_rules = astuple(match_cand)
                v[var_name] = solver.IntVar(0, 1, var_name)
                c['gold_' + str(i)].SetCoefficient(v[var_name], 1)
                for rule_index in cand_rules:
                    c['cand_' + str(rule_index)].SetCoefficient(v[var_name], 1)
                c['min_match'].SetCoefficient(v[var_name], null_cost)
                objective.SetCoefficient(v[var_name], cost)
                size_cnt[len(cand_rules)] += 1

            # update the state and continue onto the next block in gold
            curr_state = gold_state

    if not silent:
        print('Counts for all top power set sizes', size_cnt)
        print(f'Number of action gold blocks: {number_active_gold_blocks}')

    # we now update min_match with bounds based on the number of actually active gold blocks
    # min_match_number = int(match_proportion * number_active_gold_blocks)
    # c['min_match'].SetBounds(min_match_number, number_active_gold_blocks)
    c['min_match'].SetBounds(match_proportion * total_null_costs, total_null_costs)

    # solve the ILP
    objective.SetMinimization()
    if not silent:
        print("Solving the ILP...")
    status = solver.Solve()

    # reconstruct the solution and return it
    final_value = solver.Objective().Value()
    max_cost = total_null_costs
    if not null_only:
        _, _, max_cost, _, _ = match_rulesets(gold,
                                              cand,
                                              env,
                                              match_proportion,
                                              k_matches,
                                              max_power_set_size,
                                              use_greedy_growth,
                                              interpret_matching,
                                              silent,
                                              null_only=True)

    if not silent:
        print('Minimum objective function value = %f' % final_value)
        print('Minimum objective function value percentage = %f' % (1.0 - final_value / max_cost))

    # interpret solution as a matching, returning a list pairing indices of blocks in gold to a list of indices of matched rules in cand
    matching = []
    for name, var in v.items():
        if var.solution_value():  # ie if this variable was set to 1
            # print('%s = %d' % (var.name(), var.solution_value()))

            # process the variable name to extract the IDs of the matched rules
            # example name: b_16,(20,24,27)
            id_half = name.split('_')[1]  # name: 16,(20,24,27)
            gold_half, cand_half = id_half.split('(')  # 16, ; 20,24,27)
            gold_var = int(gold_half[:-1])  # remove the comma and turn into an int
            cand_vars = cand_half[:-1].split(',')  # remove the right paren and split on the commas
            cand_vars = [int(x) for x in cand_vars]  # make the numbers into ints

            match = [gold_var, cand_vars]
            matching.append(match)

            if interpret_matching:
                gold_id, cand_ids = match
                gold_block = gold[gold_id]
                cand_rules = [cand[j] if j > -1 else None for j in cand_ids]
                cost = objective.GetCoefficient(v[name])
                print('---')
                print('gold block', gold_id, ':', gold_block)
                print('matched to rules:', cand_rules)
                print('with dist', str(cost))

    return matching, status, final_value, max_cost, size_cnt


if __name__ == "__main__":
    add_argument("match_proportion", dtype=float, default=.7, msg="Proportion of gold blocks to force matches on")
    add_argument("k_matches", dtype=int, default=10, msg="Number of matches to consider per gold block")
    add_argument("interpret_matching", dtype=bool, default=False, msg="Flag to print out the rule matching")
    add_argument('cand_path', dtype=str, default='data/toy_cand_rules.txt', msg='Path to the candidate rule file.')
    add_argument('out_path', dtype=str, msg='File to write the results to.')
    add_argument('max_power_set_size', dtype=int, default=3, msg='Maximum power set size.')
    add_argument("use_greedy_growth", dtype=bool, default=False, msg="Flag to grow the kept candidates greedily.")
    add_argument("silent", dtype=bool, default=False, msg="Flag to suppress printing.")
    add_argument('cand_length', dtype=int, default=0, msg='Only take the first n candidate rules if positive.')

    manager, gold, states, refs = rule.simulate()
    initial_state = states[0]

    cand = read_rules_from_txt(g.cand_path)
    if g.cand_length > 0:
        cand = cand[:g.cand_length]
    # gold = read_rules_from_txt('data/toy_gold_rules.txt')

    # turn gold rules into singleton lists since we expect gold to be in the form of blocks
    # Group rules by refs. Assume refs are chronologically ordered.
    gold_blocks = list()
    ref_set = set()  # This stores every ref that has been encountered.
    for gold_rule, ref in zip(gold, refs):
        if ref not in ref_set:
            ref_set.add(ref)
            gold_blocks.append([gold_rule])
        else:
            gold_blocks[-1].append(gold_rule)

    env = manager.env

    results = match_rulesets(gold_blocks, cand, env,
                             g.match_proportion, g.k_matches, g.max_power_set_size, g.use_greedy_growth, g.interpret_matching, g.silent)
    if g.out_path:
        with open(g.out_path, 'wb') as fout:
            pickle.dump(results, fout)

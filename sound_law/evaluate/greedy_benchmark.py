from __future__ import annotations

import logging
import pickle
import re
import bisect
import random
import string
from dataclasses import dataclass, field
from typing import ClassVar, List, Set, Dict, Optional, Union
import pandas as pd

# from dev_misc import add_argument, g
# from sound_law.main import setup
# from sound_law.rl.action import SoundChangeAction
# import sound_law.rl.rule as rule
# from sound_law.rl.rule import HandwrittenRule

# from .ilp import match_rulesets

class ToyEnv():
    # in this toy environment, actions are single letters and states are strings created by appending them

    def __init__(self, start_state, end_state):
        self.start = start_state
        self.end = end_state

    def apply_action(self, state, act):
        # somehow apply action to state
        new_state = state + act
        return new_state

    def apply_block(self, state, block):
        '''Applies a block of actions in order'''
        for act in block:
            state = self.apply_action(state, act)
        return state

    def get_state_edit_dist(self, state1, state2):
        # somehow compute the edit distance between these two states
        state1_value = sum((ord(char) for char in state1), 0.0)
        state2_value = sum((ord(char) for char in state2), 0.0)
        return state1_value - state2_value

def get_possible_actions(state: VocabState) -> List[SoundChangeAction]:
    # toy function for now
    return [char for char in string.ascii_lowercase]


def greedily_find_rules(env: SoundChangeEnv, n_rules: int) -> List[SoundChangeAction]:
    '''Greedily finds the n best rules evolving in a certain SoundChangeEnv, where best is greedily defined as decreasing the edit distance the most'''
    curr_state = env.start
    end_state = env.end
    chosen_rules = []

    def dist_from_end(state: VocabState) -> float:
        return env.get_state_edit_dist(state, end_state)

    for _ in range(n_rules):
        best_distance = None
        best_act = None
        # find all viable rules
        possible_actions = get_possible_actions(curr_state)
        for act in possible_actions:
            # evaluate this rule
            act_state = env.apply_action(curr_state, act)
            act_dist = dist_from_end(act_state)
            # update best distance/action as necessary to find action with smallest edit distance from end state
            if best_act is None or (act_dist < best_distance):
                best_act = act
                best_distance = act_dist
        # set current state to one resulting from best rule and move on
        curr_state = env.apply_action(curr_state, best_act)
        chosen_rules.append(best_act)

    return chosen_rules


def beam_search_find_rules(env: SoundChangeEnv, n_rules: int, beam_width: int) -> List[SoundChangeAction]:
    '''Desc'''
    curr_state = env.start
    end_state = env.end
    chosen_rules = []

    def dist_from_end(state: VocabState) -> float:
        return env.get_state_edit_dist(state, end_state)

    # list of tuples of explored action lists, each tuple (distance, [actions], state)
    curr_beams = [(dist_from_end(env.start), [], env.start)] # start the beam search with the "no actions" tuple

    for _ in range(n_rules):
        new_beams = []
        for beam in curr_beams:
            beam_dist, beam_actions, beam_state = beam
            possible_actions = get_possible_actions(beam_state) # should pull this out of the for loop if it turns out this function is state-invariant
            for new_act in possible_actions:
                new_state = env.apply_action(beam_state, new_act)
                new_dist = dist_from_end(new_state)
                new_tuple = (new_dist, beam_actions + [new_act], new_state)
                
                if (len(new_beams) < beam_width) or (new_dist < new_beams[-1][0]):
                    # ie, add if we haven't hit the number of ongoing beams yet or this new action sequence results in a better state than the worst one of our considered beams
                    bisect.insort_left(new_beams, new_tuple)
                    if len(new_beams) == (beam_width + 1):
                        del new_beams[-1]
        curr_beams = new_beams
    # once we've built out our list of current beams to each be action sequences of the requested length, we return them
    return curr_beams


if __name__ == "__main__":
    # manager, gold, states, refs = rule.simulate()

    toy_env = ToyEnv('', 'cat')
    
    greedy_rules = greedily_find_rules(toy_env, 3)
    print(greedy_rules)
    # matching = match_rulesets(gold, greedy_rules, toy_env)

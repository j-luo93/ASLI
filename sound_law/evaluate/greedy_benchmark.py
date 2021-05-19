from __future__ import annotations

import logging
import pickle
import re
import bisect
from dataclasses import dataclass, field
from typing import ClassVar, List, Set, Dict, Optional, Union
import pandas as pd

from dev_misc import add_argument, g
from sound_law.main import setup
from sound_law.rl.action import SoundChangeAction
import sound_law.rl.rule as rule
from sound_law.rl.rule import HandwrittenRule

from .ilp import match_rulesets



def get_possible_actions(state: VocabState) -> List[SoundChangeAction]:
    # toy function for now using the toy data from earlier
    rules = []
    with open('data/toy_cand_rules.txt', 'r') as f:
        for line in f:
            rules.append(HandwrittenRule.from_str(line).to_action())
    return rules


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
            act_state = env.apply_action(act, curr_state)
            act_dist = dist_from_end(act_state)
            # update best distance/action as necessary to find action with smallest edit distance from end state
            if best_act is None or (act_dist < best_distance):
                best_act = act
                best_distance = act_dist
        # set current state to one resulting from best rule and move on
        curr_state = env.apply_action(best_act, curr_state)
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
                new_state = env.apply_action(new_act, beam_state)
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
    manager, gold, states, refs = rule.simulate()
    
    greedy_rules = greedily_find_rules(manager.env, 3)
    
    matching = match_rulesets(gold, greedy_rules, manager.env)

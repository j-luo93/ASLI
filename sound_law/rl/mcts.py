"""This file defines the Monte Carlo Tree Search class. Inspired by https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Set, Tuple

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution

from dev_misc import FT, add_argument, g, get_zeros
from dev_misc.utils import pad_for_log

from .action import SoundChangeAction, SoundChangeActionSpace
from .agent import AgentInputs, AgentOutputs, BasePG
from .env import SoundChangeEnv
from .trajectory import VocabState


class Mcts:

    add_argument('puct_c', default=5.0, dtype=float, msg='Exploration constant.')

    def __init__(self,
                 action_space: SoundChangeActionSpace,
                 agent: BasePG,
                 env: SoundChangeEnv,
                 end_state: VocabState):
        self.action_space = action_space
        self.agent = agent
        self.env = env
        self.end_state = end_state
        self.reset()

    def reset(self):
        self.Psa: Dict[VocabState, Distribution] = dict()  # Initial policy.
        self.Wsa: Dict[VocabState, FT] = dict()
        self.Nsa: Dict[VocabState, FT] = dict()  # State-action visit counts.

    @torch.no_grad()
    def select(self, root: VocabState) -> Tuple[VocabState, List[Tuple[VocabState, SoundChangeAction]]]:
        """Select the node to expand."""
        state = root
        path = list()
        while state != self.end_state and state in self.Psa:
            p = self.Psa[state].probs
            n_s_a = self.Nsa[state]
            n_s = n_s_a.sum()
            w = self.Wsa[state]
            q = w / (n_s_a + 1e-8)
            u = g.puct_c * p * (math.sqrt(n_s) / (1 + n_s_a))
            _, best_a = (q + u).max(dim=-1)
            action = self.action_space.get_action(best_a.item())
            path.append((state, action))
            new_state, done, reward = self.env(state, action)

            state = new_state

        logging.debug('Selected the following node:')
        logging.debug(pad_for_log(state.hash_str))

        return state, path

    @torch.no_grad()
    def expand(self, state: VocabState) -> float:
        """Expand and evaluate the leaf node."""
        if state == self.end_state:
            return 1.0
        action_masks = self.action_space.get_permissible_actions(state, ret_tensor=True)
        policy = self.agent.get_policy(state, action_masks)
        value = self.agent.get_values(state)
        self.Psa[state] = policy
        self.Wsa[state] = get_zeros(len(self.action_space))
        self.Nsa[state] = get_zeros(len(self.action_space))

        return value.item()

    @torch.no_grad()
    def backup(self, path: List[Tuple[VocabState, SoundChangeAction]], value: float):
        for s, a in path:
            self.Nsa[s][a.action_id] += 1
            self.Wsa[s][a.action_id] += value

    @torch.no_grad()
    def play(self, state: VocabState) -> Tuple[Distribution, VocabState]:
        exp = self.Nsa[state].pow(0.2)
        probs = exp / (exp.sum(dim=-1, keepdims=True) + 1e-8)
        pi = Categorical(probs=probs)
        action_id = pi.sample().item()
        action = self.action_space.get_action(action_id)
        new_state, done, reward = self.env(state, action)
        return pi, new_state

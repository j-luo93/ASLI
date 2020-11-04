"""This file defines the Monte Carlo Tree Search class. Inspired by https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple, Union, overload

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from dev_misc import NDA, add_argument, g, get_tensor, get_zeros
from dev_misc.utils import pad_for_log

from .action import SoundChangeAction, SoundChangeActionSpace
from .agent import AgentInputs, AgentOutputs, BasePG
from .env import SoundChangeEnv
from .trajectory import VocabState

# from torch.distributions.distribution import Distribution

MISSING = object()


class Mcts:

    """Monte Carlo Tree Search class. Everything should be done on cpu except for evaluation.
    Use numpy arrays by default since we can potentially speed up some process through cython
    and parallel processing.
    """

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
        self.Psa: Dict[VocabState, NDA] = dict()  # Initial policy.
        self.Wsa: Dict[VocabState, NDA] = dict()
        self.Nsa: Dict[VocabState, NDA] = dict()  # State-action visit counts.

    @torch.no_grad()
    @profile
    def select(self, root: VocabState, depth_limit: int) -> Tuple[VocabState, List[Tuple[VocabState, SoundChangeAction]]]:
        """Select the node to expand."""
        state = root
        last_state = action = None
        path = list()
        while state != self.end_state and state in self.Psa:
            p = self.Psa[state]
            n_s_a = self.Nsa[state]
            n_s = n_s_a.sum()
            w = self.Wsa[state]
            q = w / (n_s_a + 1e-8)
            u = g.puct_c * p * (math.sqrt(n_s) / (1 + n_s_a))

            action_masks = self.action_space.get_permissible_actions(state, return_type='numpy')
            scores = np.where(action_masks, q + u, np.full_like(q, -9999.9, dtype='float32'))
            best_a = scores.argmax(axis=-1)

            action = self.action_space.get_action(best_a)
            path.append((state, action))
            new_state, done, reward = self.env(state, action)

            last_state, state = state, new_state
            depth_limit -= 1
            if depth_limit <= 0:
                break

        # self.on = True
        # if self.on:
        #     if last_state is None:
        #         logging.debug(f'Selected the root (id {state.s_id}):')
        #         logging.debug(pad_for_log(state.s_key))
        #     else:
        #         logging.debug(f'Path {[state.s_id for state, _ in path]}')
        #         logging.debug(
        #             f'Selected the following node (id {state.s_id} from {last_state.s_id}, action {action.action_id}):')
        #         to_log = str(action) + '\n\n'
        #         paddings = max(map(len, last_state.s_key.split('\n'))) + 8
        #         for w0, w1 in zip(last_state.s_key.split('\n'), state.s_key.split('\n')):
        #             to_log += w0 + ' ' * (paddings - len(w0)) + w1 + '\n'
        #         logging.debug(pad_for_log(to_log.strip()))
        #     import time; time.sleep(0.1)

        return state, path

    @overload
    def expand(self, state: VocabState) -> float: ...

    @overload
    def expand(self, states: List[VocabState]) -> List[float]: ...

    @torch.no_grad()
    # @profile
    def expand(self, states):
        """Expand and evaluate the leaf node."""
        ret_lst = True
        if isinstance(states, VocabState):
            states = [states]
            ret_lst = False

        values = [None] * len(states)
        outstanding_idx = list()
        outstanding_states = list()
        # Deal with end states first.
        for i, state in enumerate(states):
            if state == self.end_state:
                values[i] = 1.0
            else:
                outstanding_idx.append(i)
                outstanding_states.append(state)

        # Collect states that need evaluation.
        # FIXME(j_luo) Add no_grad
        if outstanding_states:
            action_masks = [
                self.action_space.get_permissible_actions(state, return_type='tensor')
                for state in outstanding_states
            ]
            action_masks = torch.stack(action_masks, new_name='batch').align_to('batch', 'action')
            ids = get_tensor(np.stack([state.ids for state in outstanding_states], axis=0))
            ids.rename_('batch', 'pos', 'word')
            # FIXME(j_luo) Maybe gpu is still faster?
            probs = self.agent.get_policy(ids, action_masks).probs.cpu().numpy()
            agent_values = self.agent.get_values(ids).cpu().numpy()

            for i, state, p, v in zip(outstanding_idx, outstanding_states, probs, agent_values):
                # NOTE(j_luo) Values should be returned even if states are duplicates or have been visited.
                values[i] = v
                # NOTE(j_luo) Skip duplicate states (due to exploration collapse) or visited states (due to rollout truncation).
                if state in self.Psa:
                    continue

                self.Psa[state] = p
                self.Wsa[state] = np.zeros([len(self.action_space)])
                self.Nsa[state] = np.zeros([len(self.action_space)])

        if ret_lst:
            return values
        return values[0]

        # action_masks = self.action_space.get_permissible_actions(state, return_type='tensor')
        # policy = self.agent.get_policy(state, action_masks)
        # value = self.agent.get_values(state)
        # self.Psa[state] = policy.probs.cpu().numpy()  # FIXME(j_luo) Maybe gpu is still faster?
        # self.Wsa[state] = np.zeros([len(self.action_space)])
        # self.Nsa[state] = np.zeros([len(self.action_space)])

        # return value.item()

    @torch.no_grad()
    @profile
    def backup(self,
               path: List[Tuple[VocabState, SoundChangeAction]],
               value: Optional[float] = None,
               complete: Optional[bool] = MISSING):
        if g.use_wu_uct and complete is MISSING:
            raise ValueError(f'Missing `incomplete` value for WU-UCT.')

        for s, a in path:
            if not g.use_wu_uct or not complete:
                self.Nsa[s][a.action_id] += 1
            if not g.use_wu_uct or complete:
                self.Wsa[s][a.action_id] += value

    @torch.no_grad()
    @profile
    def play(self, state: VocabState) -> Tuple[NDA, SoundChangeAction, VocabState]:
        exp = np.power(self.Nsa[state], 1.0)
        probs = exp / (exp.sum(axis=-1, keepdims=True) + 1e-8)
        action_id = np.random.choice(range(len(probs)), p=probs)
        action = self.action_space.get_action(action_id)
        new_state, done, reward = self.env(state, action)
        return probs, action, new_state

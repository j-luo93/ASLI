"""This file defines the Monte Carlo Tree Search class. Inspired by https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union, overload

import numpy as np
import torch

from dev_misc import NDA, add_argument, g, get_tensor, get_zeros
from dev_misc.utils import pad_for_log

from .action import SoundChangeAction, SoundChangeActionSpace
from .agent import AgentInputs, AgentOutputs, BasePG
from .env import SoundChangeEnv, stack_ids
from .mcts_fast import parallel_select  # pylint: disable=no-name-in-module
from .trajectory import VocabState


class Mcts:

    """Monte Carlo Tree Search class. Everything should be done on cpu except for evaluation.
    Use numpy arrays by default since we can potentially speed up some process through cython
    and parallel processing.
    """

    add_argument('puct_c', default=5.0, dtype=float, msg='Exploration constant.')
    add_argument('virtual_loss', default=1.0, dtype=float, msg='Virtual loss per game.')
    add_argument('game_count', default=3, dtype=int, msg='How many virtual games lost.')
    add_argument('mixing', default=0.5, dtype=float, msg='Mixing lambda hyperparameter.')
    add_argument('num_workers', default=4, dtype=int, msg='Number of workers for parallelizing MCTS.')

    def __init__(self,
                 action_space: SoundChangeActionSpace,
                 agent: BasePG,
                 env: SoundChangeEnv,
                 end_state: VocabState):
        self.action_space = action_space
        self.agent = agent
        self.env = env
        self.end_state = end_state

        # NOTE(j_luo) This keeps track all selected states in history.
        self._states: List[VocabState] = list()
        self.reset()

    def reset(self):
        for s in self._states:
            s.reset()

    # @profile
    def parallel_select(self, root: VocabState, num_sims: int, depth_limit: int) -> List[VocabState]:
        return parallel_select(root, self.end_state, self.action_space, self.env,
                               num_sims, g.num_workers, depth_limit,
                               g.puct_c, g.game_count, g.virtual_loss)

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
        if outstanding_states:
            action_masks = [
                self.action_space.get_action_mask(state)
                for state in outstanding_states
            ]
            am_tensor = get_tensor(np.stack(action_masks, axis=0)).rename('batch', 'action')
            id_seqs = list()
            for state in outstanding_states:
                id_seqs.extend(state.vocab)
            id_seqs = stack_ids(id_seqs, am_tensor.size('batch'), len(state))
            probs = self.agent.get_policy(id_seqs, am_tensor).probs.cpu().numpy()
            agent_values = self.agent.get_values(id_seqs).cpu().numpy()

            for i, state, p, v, am in zip(outstanding_idx, outstanding_states, probs, agent_values, action_masks):
                # NOTE(j_luo) Values should be returned even if states are duplicates or have been visited.
                values[i] = v
                # NOTE(j_luo) Skip duplicate states (due to exploration collapse) or visited states (due to rollout truncation).
                if not state.is_leaf():
                    continue

                self._states.append(state)

                # See issue here https://github.com/cython/cython/issues/2204. Memoryview with bool dtype is still not supported.
                state.expand(p, am.astype('uint8'))

        if ret_lst:
            return values
        return values[0]

    # @profile
    def backup(self,
               state: VocabState,
               value: float):
        state.backup(value, g.mixing, g.game_count, g.virtual_loss)

    # @profile
    def play(self, state: VocabState) -> Tuple[NDA, SoundChangeAction, VocabState]:
        exp = np.power(state.action_count, 1.0)
        probs = exp / (exp.sum(axis=-1, keepdims=True) + 1e-8)
        action_id = np.random.choice(range(len(probs)), p=probs)
        action = self.action_space.get_action(action_id)
        new_state, done, reward = self.env(state, action)
        # Set `state.played` to True. This would prevent future backups from going further up.
        state.play()
        return probs, action, new_state

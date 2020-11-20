"""This file defines the Monte Carlo Tree Search class. Inspired by https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py.
"""

from __future__ import annotations
from dev_misc.trainlib import Tracker
from dev_misc import LT

import logging
from typing import Dict, List, Optional, Set, Tuple, Union, overload

import numpy as np
import torch

from dev_misc import NDA, add_argument, g, get_tensor, get_zeros
from dev_misc.utils import ScopedCache, pad_for_log

from .action import SoundChangeAction, SoundChangeActionSpace
from .agent import AgentInputs, AgentOutputs, BasePG
from .env import SoundChangeEnv
from .mcts_fast import (  # pylint: disable=no-name-in-module
    parallel_get_sparse_action_masks, parallel_select, parallel_stack_ids)
from .trajectory import VocabState, Trajectory


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
    add_argument('dirichlet_alpha', default=0.03, dtype=float, msg='Alpha value for the Dirichlet noise.')
    add_argument('noise_ratio', default=0.25, dtype=float, msg='Mixing ratio for the Dirichlet noise.')

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
        self._total_state_ids: Set[int] = set()
        self.reset()

    def reset(self):
        for s in self._states:
            s.reset()
        # logging.debug(f'Total number of states reset: {len(self._states)}.')
        self._state_ids: Set[int] = set()
        self._states: List[VocabState] = list()

    # def unplay(self):
    #     for s in self._states:
    #         s.unplay()

    # @profile
    def parallel_select(self, root: VocabState, num_sims: int, depth_limit: int) -> List[VocabState]:
        return parallel_select(root, self.end_state, self.action_space, self.env,
                               num_sims, g.num_workers, depth_limit,
                               g.puct_c, g.game_count, g.virtual_loss)

    def clear_subtree(self, state: VocabState):
        self._total_state_ids.clear()
        state.clear_subtree()

    @property
    def num_cached_states(self) -> int:
        return len(self._total_state_ids)

    @overload
    def expand(self, state: VocabState) -> float: ...

    @overload
    def expand(self, states: List[VocabState]) -> List[float]: ...

    def expand(self, states, steps: Optional[Union[int, LT]] = None):
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
                # NOTE(j_luo) This value is used for backup. If already reaching the end state, the final reward is either accounted for by the step reward, or by the value network. Therefore, we need to set it to 0.0 here.
                values[i] = 0.0
            else:
                outstanding_idx.append(i)
                outstanding_states.append(state)

        # Collect states that need evaluation.
        if outstanding_states:
            indices, action_masks, num_actions = parallel_get_sparse_action_masks(outstanding_states, g.num_workers)
            indices = get_tensor(indices)
            am_tensor = get_tensor(action_masks)
            id_seqs = parallel_stack_ids(outstanding_states, g.num_workers)
            id_seqs = get_tensor(id_seqs).rename('batch', 'pos', 'word')
            if steps is not None and not isinstance(steps, int):
                steps = steps[outstanding_idx]

            with ScopedCache('state_repr'):
                probs = self.agent.get_policy(id_seqs, am_tensor, indices=indices, sparse=True).probs.cpu().numpy()
                if g.use_value_guidance:
                    agent_values = self.agent.get_values(id_seqs, steps=steps).cpu().numpy()
                else:
                    agent_values = np.zeros([len(probs)], dtype='float32')

            for i, state, p, v, na in zip(outstanding_idx, outstanding_states, probs, agent_values, num_actions):
                # NOTE(j_luo) Values should be returned even if states are duplicates or have been visited.
                values[i] = v
                # NOTE(j_luo) Skip duplicate states (due to exploration collapse) or visited states (due to rollout truncation).
                if not state.is_leaf():
                    continue

                if state.idx not in self._state_ids:
                    self._state_ids.add(state.idx)
                    self._states.append(state)
                if state.idx not in self._total_state_ids:
                    self._total_state_ids.add(state.idx)

                # See issue here https://github.com/cython/cython/issues/2204. Memoryview with bool dtype is still not supported.
                state.expand(p[:na])

        if ret_lst:
            return values
        return values[0]

    # @profile
    def backup(self,
               state: VocabState,
               value: float):
        state.backup(value, g.mixing, g.game_count, g.virtual_loss)

    # @profile
    def play(self, state: VocabState) -> Tuple[NDA, SoundChangeAction, float, VocabState]:
        exp = np.power(state.action_count, 1.0)
        probs = exp / (exp.sum(axis=-1, keepdims=True) + 1e-8)
        if g.use_max_value:
            best_i = np.argmax(state.max_value)
        else:
            best_i = np.random.choice(range(len(probs)), p=probs)
        action_id = state.action_allowed[best_i]
        action = self.action_space.get_action(action_id)
        new_state, done, reward = self.env(state, best_i, action)
        # Set `state.played` to True. This would prevent future backups from going further up.
        state.play()
        return probs, action, reward, new_state

    def add_noise(self, state: VocabState):
        """Add Dirichlet noise to `state`, usually the root."""
        num_actions = state.get_num_allowed()
        noise = np.random.dirichlet(g.dirichlet_alpha * np.ones(num_actions)).astype('float32')
        state.add_noise(noise, g.noise_ratio)

    def collect_episodes(self, init_state: VocabState, end_state: VocabState, tracker: Tracker) -> List[Trajectory]:
        logging.info(f'{self.num_cached_states} states cached.')
        logging.info(f'{self.env.action_space.cache_size} words cached.')
        if self.num_cached_states > 300000:
            logging.info(f'Clearing up all the tree nodes.')
            self.clear_subtree(init_state)
        if self.env.action_space.cache_size > 300000:
            logging.info(f'Clearing up all the cached words.')
            self.env.action_space.clear_cache()
        trajectories = list()
        self.agent.eval()
        with self.agent.policy_grad(False), self.agent.value_grad(False):
            for ei in range(g.num_episodes):
                root = init_state
                self.reset()
                steps = 0 if g.use_finite_horizon else None
                value = self.expand(root, steps=steps)
                self.backup(root, value)

                trajectory = Trajectory(root, end_state)
                # Episodes have max rollout length.
                for ri in range(g.max_rollout_length):
                    self.add_noise(root)
                    depth_limit = g.max_rollout_length - ri
                    # Run many simulations before take one action. Simulations take place in batches. Each batch
                    # would be evaluated and expanded after batched selection.
                    num_batches = g.num_mcts_sims // g.expansion_batch_size
                    for _ in range(num_batches):
                        new_states, steps_left = self.parallel_select(root, g.expansion_batch_size, depth_limit)
                        steps = g.max_rollout_length - get_tensor(steps_left) if g.use_finite_horizon else None
                        values = self.expand(new_states, steps=steps)
                        for state, value in zip(new_states, values):
                            self.backup(state, value)
                        tracker.update('mcts', incr=g.expansion_batch_size)
                    probs, action, reward, new_state = self.play(root)
                    trajectory.append(action, new_state, new_state.done, reward, mcts_pi=probs)
                    if ri == 0 and ei % g.episode_check_interval == 0:
                        logging.debug(pad_for_log(str(get_tensor(root.action_count).topk(20))))
                        logging.debug(pad_for_log(str(get_tensor(root.q).topk(20))))
                    root = new_state

                    tracker.update('rollout')
                    if root.done:
                        break
                if ei % g.episode_check_interval == 0:
                    out = ', '.join(f'({edge.a}, {edge.r:.3f})' for edge in trajectory)
                    logging.debug(pad_for_log(out))

                trajectories.append(trajectory)
                tracker.update('episode')

        return trajectories

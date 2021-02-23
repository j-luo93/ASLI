"""This file defines the Monte Carlo Tree Search class. Inspired by https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Union, overload

import numpy as np
import torch

from dev_misc import LT, NDA, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Tracker
from dev_misc.utils import ScopedCache, pad_for_log
from sound_law.rl.action import SoundChangeAction
from sound_law.rl.agent import AgentInputs, AgentOutputs, BasePG
from sound_law.rl.env import SoundChangeEnv
from sound_law.rl.trajectory import Trajectory, VocabState

# pylint: disable=no-name-in-module
from .mcts_cpp import PyMcts, parallel_stack_ids

# pylint: enable=no-name-in-module


class Mcts(PyMcts):

    """Monte Carlo Tree Search class. Everything should be done on cpu except for evaluation.
    Use numpy arrays by default since we can potentially speed up some process through cython
    and parallel processing.
    """

    add_argument('puct_c', default=5.0, dtype=float, msg='Exploration constant.')
    add_argument('virtual_loss', default=1.0, dtype=float, msg='Virtual loss per game.')
    add_argument('game_count', default=3, dtype=int, msg='How many virtual games lost.')
    add_argument('heur_c', default=1.0, dtype=float, msg='Heuristic constant.')
    add_argument('mixing', default=0.5, dtype=float, msg='Mixing lambda hyperparameter.')
    add_argument('num_workers', default=4, dtype=int, msg='Number of workers for parallelizing MCTS.')
    add_argument('dirichlet_alpha', default=0.03, dtype=float, msg='Alpha value for the Dirichlet noise.')
    add_argument('noise_ratio', default=0.25, dtype=float, msg='Mixing ratio for the Dirichlet noise.')

    # def __init__(self,
    #              env: SoundChangeEnv,
    #              puct_c: float,
    #              game_count: int,
    #              virtual_loss: float,
    #              num_threads: int,
    #              agent: BasePG):
    #     self.env = env
    #     self.agent = agent
    #     # NOTE(j_luo) This keeps track all selected states in history.
    #     # self._states: List[VocabState] = list()
    #     # self._total_state_ids: Set[int] = set()
    #     # self.reset()
    def __init__(self, *args, agent: BasePG = None, **kwargs):
        self.agent = agent

    def reset(self, evict: bool = True):
        # for s in self._states:
        #     s.reset()
        # logging.debug(f'Total number of states reset: {len(self._states)}.')
        # self._state_ids: Set[int] = set()
        # self._states: List[VocabState] = list()
        # self.env.prune(self.env.start)
        self.env.clear_priors(self.env.start, True)
        self.env.clear_stats(self.env.start, True)
        # logging.info(f'#words: {self.env.num_words}')
        # num_desc = self.env.start.num_descendants
        # logging.info(f'#nodes: {num_desc}')
        # if evict and num_desc > 1000000:
        self.env.evict(1000000)

    def evaluate(self, states, steps: Optional[Union[int, LT]] = None) -> List[float]:
        """Expand and evaluate the leaf node."""
        values = [None] * len(states)
        outstanding_idx = list()
        outstanding_states = list()
        # Deal with end states first.
        for i, state in enumerate(states):
            if state.stopped or state.done:
                # NOTE(j_luo) This value is used for backup. If already reaching the end state, the final reward is either accounted for by the step reward, or by the value network. Therefore, we need to set it to 0.0 here.
                values[i] = 0.0
            else:
                outstanding_idx.append(i)
                outstanding_states.append(state)

        # Collect states that need evaluation.
        if outstanding_states:
            # indices, action_masks, num_actions = parallel_get_sparse_action_masks(outstanding_states, g.num_workers)
            # indices = get_tensor(indices)
            # am_tensor = get_tensor(action_masks)
            # FIXME(j_luo) maybe transpose it first???
            id_seqs = parallel_stack_ids(outstanding_states, g.num_workers)
            id_seqs = get_tensor(id_seqs).rename('batch', 'word', 'pos')
            if steps is not None and not isinstance(steps, int):
                steps = steps[outstanding_idx]

            # TODO(j_luo) Scoped might be wrong here.
            # with ScopedCache('state_repr'):
            # NOTE(j_luo) Don't forget to call exp().
            priors = self.agent.get_policy(id_seqs).exp()
            with NoName(priors):
                meta_priors = priors[:, [0, 2, 3, 4, 5, 6]].cpu().numpy()
                special_priors = priors[:, 1].cpu().numpy()
            if g.use_value_guidance:
                agent_values = self.agent.get_values(id_seqs, steps=steps).cpu().numpy()
            else:
                agent_values = np.zeros([len(id_seqs)], dtype='float32')

            for i, state, mp, sp, v in zip(outstanding_idx, outstanding_states, meta_priors, special_priors, agent_values):
                # NOTE(j_luo) Values should be returned even if states are duplicates or have been visited.
                values[i] = v
                # NOTE(j_luo) Skip duplicate states (due to exploration collapse) or visited states (due to rollout truncation).
                if not state.is_leaf():
                    continue

                # if state.idx not in self._state_ids:
                #     self._state_ids.add(state.idx)
                #     self._states.append(state)
                # if state.idx not in self._total_state_ids:
                #     self._total_state_ids.add(state.idx)

                self.env.evaluate(state, mp, sp)
        return values

    # def backup(self,
    #            state: VocabState,
    #            value: float):
    #     state.backup(value, g.mixing, g.game_count, g.virtual_loss)

    # def play(self, state: VocabState) -> Tuple[NDA, SoundChangeAction, float, VocabState]:
    #     exp = np.power(state.action_counts.astype('float32'), 1.0)
    #     probs = exp / (exp.sum(axis=-1, keepdims=True) + 1e-8)
    #     # if g.use_max_value:
    #     #     best_i = np.argmax(state.max_value)
    #     # else:
    #     #     best_i = np.random.choice(range(len(probs)), p=probs)
    #     # action_id = state.action_allowed[best_i]
    #     # action = self.action_space.get_action(action_id)
    #     # new_state, done, reward = self.env(state, best_i, action)
    #     # # Set `state.played` to True. This would prevent future backups from going further up.
    #     # state.play()
    #     # return probs, action, reward, new_state
    #     # best_i = state.max_index
    #     # # This means stop action.
    #     # if best_i == -1:
    #     #     action_id = 0
    #     #     action = self.env.action_space.get_action(0)
    #     #     # FIXME(j_luo) new state is wrong.
    #     #     new_state = None
    #     #     reward = 0.0
    #     # else:
    #     #     action_id = state.max_action_id
    #     #     action = self.env.action_space.get_action(action_id)
    #     #     new_state, reward = self.env.step(state, best_i, action_id)
    #     # new_state, reward, action_path = super().play(state)
    #     # action = SoundChangeAction(action_path[0], action_path[2], action_path[3], action_path[4], action_path[5], action_path[6],
    #     #                            special_type=int2st[action_path[1]])
    #     # return probs, action, reward, new_state
    #     return super().play(state), probs

    def add_noise(self, state: VocabState):
        """Add Dirichlet noise to `state`, usually the root."""
        noise = np.random.dirichlet(g.dirichlet_alpha * np.ones(7 * len(self.env.abc))).astype('float32')
        noise = noise.reshape(7, -1)
        meta_noise = noise[:6]
        special_noise = noise[6, :6]
        self.env.add_noise(state, meta_noise, special_noise, g.noise_ratio)

    def collect_episodes(self, init_state: VocabState,
                         tracker: Optional[Tracker] = None, num_episodes: int = 0, is_eval: bool = False) -> List[Trajectory]:
        # logging.info(f'{self.num_cached_states} states cached.')

        # logging.info(f'{self.env.action_space.cache_size} words cached.')
        # logging.info(f'{len(self.action_space)} actions indexed in the action space.')
        # if self.num_cached_states > 300000:
        #     logging.info(f'Clearing up all the tree nodes.')
        # self.clear_subtree(init_state)
        # if self.env.action_space.cache_size > 300000:
        #     logging.info(f'Clearing up all the cached words.')
        #     self.env.action_space.clear_cache()
        # logging.info(
        #     f'{len(self.env.word_space.site_space)} sites, {len(self.env.word_space)} words, {init_state.get_num_descendants()} nodes in total.')
        trajectories = list()
        self.agent.eval()
        num_episodes = num_episodes or g.num_episodes
        # self.env.evict(100000)
        with self.agent.policy_grad(False), self.agent.value_grad(False):
            # self.enable_timer()
            for ei in range(num_episodes):
                root = init_state
                self.reset(evict=True)
                steps = 0 if g.use_finite_horizon else None
                # self.env.action_space.set_action_allowed(root)
                self.evaluate([root], steps=steps)
                # self.backup([root], [value])

                # trajectory = Trajectory(root, end_state)
                # trajectory = Trajectory()
                # Episodes have max rollout length.
                for ri in range(g.max_rollout_length):
                    # if ri == 0:
                    #     self.enable_timer()

                    if not is_eval:
                        self.add_noise(root)
                    # Run many simulations before take one action. Simulations take place in batches. Each batch
                    # would be evaluated and expanded after batched selection.
                    # print('1')
                    num_batches = g.num_mcts_sims // g.expansion_batch_size
                    for _ in range(num_batches):
                        paths, steps = self.select(root, g.expansion_batch_size, ri, g.max_rollout_length)
                        steps = get_tensor(steps) if g.use_finite_horizon else None
                        values = self.evaluate(paths, steps=steps)
                        self.backup(paths, values)
                        # backed_up_idx = set()
                        # for state, value in zip(new_states, values):
                        #     if state.idx not in backed_up_idx:
                        #         self.backup(state, value)
                        #         backed_up_idx.add(state.idx)
                        if tracker is not None:
                            tracker.update('mcts', incr=g.expansion_batch_size)
                    if ri == 0 and ei % g.episode_check_interval == 0:
                        k = min(20, root.num_actions)
                        logging.debug(pad_for_log(str(get_tensor(root.action_counts).topk(k))))
                        logging.debug(pad_for_log(str(get_tensor(root.q).topk(k))))
                    # print('2')
                    # probs, action, reward, new_state = self.play(root)
                    # new_state = self.play(root)
                    # trajectory.append(action, new_state, reward, mcts_pi=probs)
                    # root = new_state
                    root = self.play(root)

                    # print('3')
                    if tracker is not None:
                        tracker.update('rollout')
                    if root.stopped or root.done:
                        break
                    # self.show_stats()
                trajectory = Trajectory(root)
                if ei % g.episode_check_interval == 0:
                    logging.debug(pad_for_log(str(trajectory)))

                trajectories.append(trajectory)
                if tracker is not None:
                    tracker.update('episode')
                # orig_size = init_state.clear_cache(0.2)
                # new_size = init_state.get_num_descendants()
                # logging.info(f'Clearing node cache {orig_size} -> {new_size}.')

        return trajectories

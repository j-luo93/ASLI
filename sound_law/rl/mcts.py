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
from .mcts_cpp import PyMcts, PyPS_MAX, PyPS_SAMPLE_AC, parallel_stack_ids

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
    add_argument('play_strategy', default='max', dtype=str, choices=['max', 'sample_ac'], msg='Play strategy.')

    def __init__(self, *args, agent: BasePG = None, **kwargs):
        self.agent = agent
        if g.play_strategy == 'max':
            self.play_strategy = PyPS_MAX
        else:
            self.play_strategy = PyPS_SAMPLE_AC

    def reset(self):
        # Clear priors first and then stats -- stats are needed to speed up clearing.
        self.env.clear_priors(self.env.start, True)
        self.env.clear_stats(self.env.start, True)
        logging.debug(f'#trie nodes {self.env.evict(500000)}')

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
            almts1 = almts2 = None
            if g.use_alignment:
                id_seqs, almts1, almts2 = parallel_stack_ids(
                    outstanding_states, g.num_workers, True, self.env.max_end_length)
                almts1 = get_tensor(almts1).rename('batch', 'word', 'pos')
                almts2 = get_tensor(almts2).rename('batch', 'word', 'pos')
            else:
                id_seqs = parallel_stack_ids(outstanding_states, g.num_workers, False, self.env.max_end_length)
            id_seqs = get_tensor(id_seqs).rename('batch', 'word', 'pos')
            if steps is not None and not isinstance(steps, int):
                steps = steps[outstanding_idx]

            # TODO(j_luo) Scoped might be wrong here.
            # with ScopedCache('state_repr'):
            # NOTE(j_luo) Don't forget to call exp().
            priors = self.agent.get_policy(id_seqs, almts=(almts1, almts2)).exp()
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

                self.env.evaluate(state, mp, sp)
        return values

    def add_noise(self, state: VocabState):
        """Add Dirichlet noise to `state`, usually the root."""
        noise = np.random.dirichlet(g.dirichlet_alpha * np.ones(7 * len(self.env.abc))).astype('float32')
        noise = noise.reshape(7, -1)
        meta_noise = noise[:6]
        special_noise = noise[6, :6]
        self.env.add_noise(state, meta_noise, special_noise, g.noise_ratio)

    def collect_episodes(self, init_state: VocabState,
                         tracker: Optional[Tracker] = None, num_episodes: int = 0, is_eval: bool = False) -> List[Trajectory]:
        trajectories = list()
        self.agent.eval()
        num_episodes = num_episodes or g.num_episodes
        with self.agent.policy_grad(False), self.agent.value_grad(False):
            for ei in range(num_episodes):
                root = init_state
                self.reset()
                steps = 0 if g.use_finite_horizon else None
                self.evaluate([root], steps=steps)

                # Episodes have max rollout length.
                played_path = None
                for ri in range(g.max_rollout_length):
                    if not is_eval:
                        self.add_noise(root)
                    # Run many simulations before take one action. Simulations take place in batches. Each batch
                    # would be evaluated and expanded after batched selection.
                    num_batches = g.num_mcts_sims // g.expansion_batch_size
                    for _ in range(num_batches):
                        paths, steps = self.select(root, g.expansion_batch_size, ri, g.max_rollout_length, played_path)
                        steps = get_tensor(steps) if g.use_finite_horizon else None
                        new_states = [path.get_last_node() for path in paths]
                        values = self.evaluate(new_states, steps=steps)
                        self.backup(paths, values)
                        if tracker is not None:
                            tracker.update('mcts', incr=g.expansion_batch_size)
                    if ri == 0 and ei % g.episode_check_interval == 0:
                        k = min(20, root.num_actions)
                        logging.debug(pad_for_log(str(get_tensor(root.action_counts).topk(k))))
                        logging.debug(pad_for_log(str(get_tensor(root.q).topk(k))))
                    ps = PyPS_MAX if is_eval else self.play_strategy
                    new_path = self.play(root, ri, ps)
                    if played_path is None:
                        played_path = new_path
                    else:
                        played_path.merge(new_path)
                    root = played_path.get_last_node()

                    # print('3')
                    if tracker is not None:
                        tracker.update('rollout')
                    if root.stopped or root.done:
                        break
                    # self.show_stats()
                trajectory = Trajectory(played_path, self.env.max_end_length)
                if ei % g.episode_check_interval == 0:
                    logging.debug(pad_for_log(str(trajectory)))

                trajectories.append(trajectory)
                if tracker is not None:
                    tracker.update('episode')

        return trajectories

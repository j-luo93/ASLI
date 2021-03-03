"""This file defines the environment and the collector used in the environment to collect samples.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dev_misc import FT, LT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib import pad_to_dense
from dev_misc.devlib.named_tensor import NoName
from dev_misc.utils import handle_sequence_inputs
# from sound_law.data.alphabet import PAD_ID

from .action import SoundChangeAction, SoundChangeActionSpace
from .agent import AgentInputs, VanillaPolicyGradient
from .mcts_cpp import PyEnv, PyWordSpace  # pylint: disable=no-name-in-module
from .trajectory import Trajectory, VocabState


class SoundChangeEnv(nn.Module, PyEnv):

    tnode_cls = VocabState

    add_argument(f'final_reward', default=1.0, dtype=float, msg='Final reward for reaching the end.')
    add_argument(f'step_penalty', default=0.02, dtype=float, msg='Penalty for each step if not the end state.')

    # pylint: disable=unused-argument
    def __init__(self,
                 action_space: SoundChangeActionSpace,
                 word_space: PyWordSpace,
                 s_arr, s_lengths,
                 e_arr, e_lengths,
                 final_reward: float,
                 step_penalty: float):
        nn.Module.__init__(self)

    def forward(self, state: VocabState, best_i: int, action: SoundChangeAction) -> Tuple[VocabState, bool, float]:
        return self.step(state, best_i, action)

    def show_path(self, state: VocabState) -> str:
        out = list()
        for action_id, reward in state.get_path():
            action = self.action_space.get_action(action_id)
            out.append(f'{action}, {reward:.3f}')
        return '(' + ', '.join(out) + ')'

class ToyEnv(SoundChangeEnv):
    import random

    def __init__(self, init_state):
        self.init_state = init_state

    def apply_action(self, act, state):
        # somehow apply action to state
        new_state = state
        return new_state
    
    def apply_block(self, block, state):
        '''Applies a block of actions in order'''
        for act in block:
            state = self.apply_action(act, state)
        return state

    def dist_between(self, state1, state2):
        # somehow compute the edit distance between these two states
        return random.random() * random.randint(1, 20)

    def compare_effects(self, act1, act2, state):
        state1 = self.apply_action(act1, state)
        state2 = self.apply_action(act2, state)
        return self.dist_between(state1, state2)


class TrajectoryCollector:
    """This collects trajectories and (flattened/batched) samples."""

    def __init__(self,
                 max_sample_size: int,
                 max_rollout_length: Optional[int] = None,
                 truncate_last: bool = False):
        self._max_sample_size = max_sample_size
        self._max_rollout_length = max_rollout_length
        # Whether to truncate the last trajectory if enough samples have been collected.
        self._truncate_last = truncate_last

    @torch.no_grad()
    def collect(self,
                agent: VanillaPolicyGradient,
                env: SoundChangeEnv,
                init_state: VocabState,
                end_state: VocabState) -> AgentInputs:
        """Collect a batch of states, actions and rewards."""
        # Collect in eval mode.
        agent.eval()

        def get_new_trajectory() -> Trajectory:
            return Trajectory(init_state, end_state)

        trajectory = get_new_trajectory()
        trajectories = [trajectory]
        n_samples = 0
        while True:
            # Whether we have collected enough samples for the last trajectory (which might not have a reasonably long action sequence).
            collected_enough_last = self._truncate_last and n_samples >= self._max_sample_size
            if collected_enough_last:
                break

            # Whether the current rollout is long enough to be truncated (regardless of whether the trajectory is done or not).
            long_rollout = self._max_rollout_length is not None and len(trajectory) >= self._max_rollout_length
            if trajectory.done or long_rollout:
                trajectory = get_new_trajectory()
                trajectories.append(trajectory)
                # Stop when we have collected enough samples (either done or with properly long rollouts).
                if n_samples >= self._max_sample_size:
                    break

            state = trajectory.latest_state

            action_masks = get_tensor(env.action_space.get_action_mask(state))
            policy = agent.get_policy(state, action_masks)
            action = agent.sample_action(policy)
            next_state, done, next_reward = env(state, action)
            trajectory.append(action, next_state, done, next_reward)
            n_samples += 1

        # Make a batch out of all the states and actions in the list of trajectories. Note that only starting states are batched.
        return AgentInputs.from_trajectories(trajectories, env.action_space)

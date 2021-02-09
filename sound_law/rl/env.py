"""This file defines the environment and the collector used in the environment to collect samples.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from dev_misc import FT, LT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib import pad_to_dense
from dev_misc.devlib.named_tensor import NoName
from dev_misc.utils import handle_sequence_inputs
from sound_law.data.alphabet import PAD_ID, Alphabet, EMP

from .action import SoundChangeAction
from .agent import AgentInputs, VanillaPolicyGradient
# pylint: disable=no-name-in-module
from .mcts_cpp import PyActionSpaceOpt, PyEnv, PyEnvOpt, PyWordSpaceOpt
# pylint: enable=no-name-in-module
from .trajectory import Trajectory, VocabState


class SoundChangeEnv(PyEnv):

    tnode_cls = VocabState

    add_argument(f'final_reward', default=1.0, dtype=float, msg='Final reward for reaching the end.')
    add_argument(f'step_penalty', default=0.02, dtype=float, msg='Penalty for each step if not the end state.')

    def register_changes(self, abc: Alphabet):
        # # Set class variable for `SoundChangeAction` here.
        SoundChangeAction.abc = abc

        # Register unconditional actions first.
        units = [u for u in abc if u not in abc.special_units]

        def register_uncondional_action(u1: str, u2: str, cl: bool = False, gb: bool = False):
            id1 = abc[u1]
            id2 = abc[u2]
            if cl:
                self.register_cl_map(id1, id2)
            elif gb:
                if u1.startswith('i'):
                    self.register_gbj(id1, id2)
                else:
                    assert u1.startswith('u')
                    self.register_gbw(id1, id2)
            else:
                self.register_permissible_change(id1, id2)

        for u1, u2 in abc.edges:
            register_uncondional_action(u1, u2)
        for u in units:
            register_uncondional_action(u, EMP)
        for u1, u2 in abc.cl_map.items():
            register_uncondional_action(u1, u2, cl=True)
        # for u1, u2 in abc.gb_map.items():
        #     register_uncondional_action(u1, u2, gb=True)

        # self.set_vowel_info(abc.vowel_mask, abc.vowel_base, abc.vowel_stress, abc.stressed_vowel, abc.unstressed_vowel)
        # self.set_glide_info(abc['j'], abc['w'])

    def forward(self, state: VocabState, best_i: int, action: SoundChangeAction) -> Tuple[VocabState, bool, float]:
        return self.step(state, best_i, action)

    def show_path(self, state: VocabState) -> str:
        out = list()
        for action_id, reward in state.get_path():
            action = self.action_space.get_action(action_id)
            out.append(f'{action}, {reward:.3f}')
        return '(' + ', '.join(out) + ')'


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

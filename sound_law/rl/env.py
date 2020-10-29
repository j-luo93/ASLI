"""This file defines the environment and the collector used in the environment to collect samples.
"""
from __future__ import annotations

from dev_misc import get_zeros
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dev_misc import FT, LT, get_tensor
from dev_misc.devlib.named_tensor import NoName
from dev_misc.utils import handle_sequence_inputs

from .action import SoundChangeAction, SoundChangeActionSpace
from .agent import AgentInputs, VanillaPolicyGradient
from .trajectory import Trajectory, VocabState


class SoundChangeEnv(nn.Module):

    def __init__(self, action_space: SoundChangeActionSpace, init_state: VocabState, end_state: VocabState):
        super().__init__()
        self.action_space = action_space
        self._init_state = init_state
        self._end_state = end_state
        self._starting_dist = init_state.dist_from(end_state)

    def forward(self, state: VocabState, action: SoundChangeAction) -> Tuple[VocabState, bool, float]:
        replace_func = handle_sequence_inputs(lambda s: s.replace(action.before, action.after))
        new_units = [replace_func(units) for units in state.units]
        new_ids = state.ids.clone()
        with NoName(new_ids):
            new_ids[new_ids == action.before_id] = action.after_id
        new_ids.rename_(*state.ids.names)
        new_state = VocabState(new_units, new_ids)
        done = new_state == self._end_state

        final_reward = 1.0 if done else 0.0
        old_dist = state.dist_from(self._end_state)
        new_dist = new_state.dist_from(self._end_state)
        incremental_reward = (old_dist - new_dist) / self._starting_dist  # * 0.02
        reward = final_reward + incremental_reward
        return new_state, done, reward


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

            permissible_actions = env.action_space.get_permissible_actions(state)
            action_ids = [action.action_id for action in permissible_actions]
            action_masks = get_zeros(len(env.action_space)).bool()
            action_masks[action_ids] = True

            policy = agent.get_policy(state, action_masks)
            action = agent.sample_action(policy)
            next_state, done, next_reward = env(state, action)
            trajectory.append(action, next_state, done, next_reward, action_masks)
            n_samples += 1

        # Make a batch out of all the states and actions in the list of trajectories. Note that only starting states are batched.
        id_seqs = list()
        next_id_seqs = list()
        action_ids = list()
        rewards = list()
        done = list()
        action_masks = list()
        for t in trajectories:
            for s0, a, s1, r, am in t:
                id_seqs.append(s0.ids)
                next_id_seqs.append(s1.ids)
                action_ids.append(a.action_id)
                rewards.append(r)
                action_masks.append(am)
            done.extend([False] * (len(t) - 1))
            done.append(t.done)
        id_seqs = torch.stack(id_seqs, new_name='batch').align_to('batch', 'pos', 'word')
        next_id_seqs = torch.stack(next_id_seqs, new_name='batch').align_to('batch', 'pos', 'word')
        action_ids = get_tensor(action_ids).rename('batch')
        rewards = get_tensor(rewards).rename('batch')
        action_masks = torch.stack(action_masks, dim=0).rename('batch', 'action')
        done = get_tensor(done).rename('batch')
        agent_inputs = AgentInputs(trajectories, id_seqs, next_id_seqs, action_ids, rewards, done, action_masks)
        return agent_inputs

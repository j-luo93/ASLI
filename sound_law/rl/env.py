"""This file defines the environment and the collector used in the environment to collect samples.
"""
from __future__ import annotations

from dev_misc.devlib import pad_to_dense
from sound_law.data.alphabet import PAD_ID
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dev_misc import FT, LT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName
from dev_misc.utils import handle_sequence_inputs

from .action import SoundChangeAction, SoundChangeActionSpace
from .agent import AgentInputs, VanillaPolicyGradient
from .mcts_fast import PyEnv  # pylint: disable=no-name-in-module
from .trajectory import Trajectory, VocabState


class SoundChangeEnv(nn.Module, PyEnv):

    add_argument(f'final_reward', default=1.0, dtype=float, msg='Final reward for reaching the end.')
    add_argument(f'step_penalty', default=0.02, dtype=float, msg='Penalty for each step if not the end state.')

    def __init__(self, init_state: VocabState, end_state: VocabState, action_space: SoundChangeActionSpace):
        nn.Module.__init__(self)
        self.init_state = init_state
        self.end_state = end_state
        self.action_space = action_space
        self._starting_dist = init_state.dist_to_end

    def forward(self, state: VocabState, action: SoundChangeAction) -> Tuple[VocabState, bool, float]:
        new_state = self.step(state, action)
        done = self.is_done(new_state)
        final_reward = g.final_reward if done else -g.step_penalty
        old_dist = state.dist_to_end
        new_dist = new_state.dist_to_end
        incremental_reward = (old_dist - new_dist) / self._starting_dist
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

            action_masks = get_tensor(env.action_space.get_action_mask(state))
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
                id_seqs.extend(s0.vocab)
                next_id_seqs.extend(s1.vocab)
                action_ids.append(a.action_id)
                rewards.append(r)
                action_masks.append(am)
            done.extend([False] * (len(t) - 1))
            done.append(t.done)
        bs = len(done)
        nw = len(init_state)

        def stack_ids(seqs):
            seqs = get_tensor(pad_to_dense(seqs, pad_idx=PAD_ID, dtype='long')[0])
            seqs = seqs.view(bs, nw, -1).rename('batch', 'word', 'pos')
            seqs = seqs.align_to('batch', 'pos', 'word')
            return seqs

        id_seqs = stack_ids(id_seqs)
        next_id_seqs = stack_ids(next_id_seqs)
        action_ids = get_tensor(action_ids).rename('batch')
        rewards = get_tensor(rewards).rename('batch')
        action_masks = torch.stack(action_masks, dim=0).rename('batch', 'action')
        done = get_tensor(done).rename('batch')
        agent_inputs = AgentInputs(trajectories, id_seqs, next_id_seqs, action_ids, rewards, done, action_masks)
        return agent_inputs

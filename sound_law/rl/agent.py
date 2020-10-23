from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from dev_misc import BT, FT, LT, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics, init_params
from dev_misc.utils import ScopedCache, cacheable
from sound_law.model.module import CharEmbedding, EmbParams, PhonoEmbedding

from .action import SoundChangeAction, SoundChangeActionSpace
from .trajectory import Trajectory, VocabState


@dataclass
class AgentInputs:
    trajectories: List[Trajectory]
    id_seqs: LT
    next_id_seqs: LT
    action_ids: LT
    rewards: FT
    done: BT

    @property
    def batch_size(self) -> int:
        return self.action_ids.size('batch')


@dataclass
class RewardOutputs:
    """This stores all relevant outputs related to rewards, including advantages and policy evaluations."""
    rtgs: FT  # rewards-to-go
    values: Optional[FT] = None
    advantages: Optional[FT] = None


@cacheable(switch='word_embedding')
def _get_word_embedding(char_emb: PhonoEmbedding, ids: LT) -> FT:
    """Get word embeddings based on ids."""
    names = ids.names + ('emb',)
    return char_emb(ids).rename(*names).mean(dim='pos')


def _get_state_repr(char_emb: PhonoEmbedding, curr_ids: LT, end_ids: LT) -> FT:
    """Get state representation used for action prediction."""
    word_repr = _get_word_embedding(char_emb, curr_ids)
    end_word_repr = _get_word_embedding(char_emb, end_ids)
    state_repr = (word_repr - end_word_repr).mean(dim='word')
    return state_repr


def _get_rewards_to_go(agent_inputs: AgentInputs) -> FT:
    rews = agent_inputs.rewards.rename(None)
    tr_lengths = get_tensor([len(tr) for tr in agent_inputs.trajectories])
    cum_lengths = tr_lengths.cumsum(dim=0)
    assert cum_lengths[-1].item() == len(rews)
    start_new = get_zeros(len(rews)).long()
    start_new.scatter_(0, cum_lengths[:-1], 1)
    which_tr = start_new.cumsum(dim=0)
    up_to_ids = cum_lengths[which_tr] - 1
    cum_rews = rews.cumsum(dim=0)
    up_to = cum_rews[up_to_ids]
    rtgs = up_to - cum_rews + rews
    return rtgs


class VanillaPolicyGradient(nn.Module):

    def __init__(self, emb_params: EmbParams, action_space: SoundChangeActionSpace, end_state: VocabState):
        super().__init__()
        self.char_emb = PhonoEmbedding.from_params(emb_params)
        self.action_space = action_space
        num_actions = len(action_space)
        self.action_predictor = nn.Linear(self.char_emb.embedding_dim, num_actions)
        self.end_state = end_state

    def get_policy(self, state_or_ids: Union[VocabState, LT]) -> Distribution:
        """Get policy distribution based on current state (and end state)."""
        if isinstance(state_or_ids, VocabState):
            ids = state_or_ids.ids
        else:
            ids = state_or_ids
        state_repr = _get_state_repr(self.char_emb, ids, self.end_state.ids)

        action_logits = self.action_predictor(state_repr)

        with NoName(action_logits):
            policy = torch.distributions.Categorical(logits=action_logits)
        return policy

    def sample_action(self, policy: Distribution) -> SoundChangeAction:
        action_id = policy.sample().item()
        return self.action_space.get_action(action_id)

    def _get_reward_outputs(self, agent_inputs: AgentInputs) -> RewardOutputs:
        """Obtain outputs related to reward."""
        rtgs = _get_rewards_to_go(agent_inputs)
        return RewardOutputs(rtgs)

    def forward(self, agent_inputs: AgentInputs) -> Tuple[FT, RewardOutputs]:
        with ScopedCache('word_embedding'):
            policy = self.get_policy(agent_inputs.id_seqs)
            with NoName(agent_inputs.action_ids):
                log_probs = policy.log_prob(agent_inputs.action_ids)
            # Compute rewards to go.
            rew_outputs = self._get_reward_outputs(agent_inputs)
        return log_probs, rew_outputs


class A2C(VanillaPolicyGradient):

    def __init__(self, emb_params: EmbParams,
                 action_space: SoundChangeActionSpace,
                 end_state: VocabState,
                 separate_emb: bool = False):
        super().__init__(emb_params, action_space, end_state)
        self.value_predictor = nn.Sequential(
            nn.Linear(self.char_emb.embedding_dim, 1),
            nn.Flatten(-2, -1))
        if separate_emb:
            self.char_emb_value = PhonoEmbedding.from_params(emb_params)
        else:
            self.char_emb_value = self.char_emb

    def get_values(self, curr_ids: LT, end_ids: LT) -> FT:
        """Get policy evaluation."""
        state_repr = _get_state_repr(self.char_emb_value, curr_ids, end_ids)
        with NoName(state_repr):
            values = self.value_predictor(state_repr)
        return values

    def _get_reward_outputs(self, agent_inputs: AgentInputs) -> RewardOutputs:
        end_ids = self.end_state.ids
        values = self.get_values(agent_inputs.id_seqs, end_ids)
        rtgs = _get_rewards_to_go(agent_inputs)
        advantages = rtgs - values.detach()
        return RewardOutputs(rtgs, values=values, advantages=advantages)

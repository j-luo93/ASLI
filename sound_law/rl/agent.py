from dataclasses import dataclass, field
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from dev_misc import FT, LT, get_tensor, get_zeros
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

    @property
    def batch_size(self) -> int:
        return self.action_ids.size('batch')


class VanillaPolicyGradient(nn.Module):

    def __init__(self, char_emb: CharEmbedding, action_space: SoundChangeActionSpace, end_state: VocabState):
        super().__init__()
        self.char_emb = char_emb
        self.action_space = action_space
        num_actions = len(action_space)
        self.action_predictor = nn.Linear(char_emb.embedding_dim, num_actions)
        self.end_state = end_state

    @cacheable(switch='word_embedding')
    def get_word_embedding(self, ids: LT) -> FT:
        """Get word embeddings based on ids."""
        names = ids.names + ('emb',)
        return self.char_emb(ids).rename(*names).mean(dim='pos')

    def get_state_repr(self, curr_ids: LT, end_ids: LT) -> FT:
        """Get state representation used for action prediction."""
        word_repr = self.get_word_embedding(curr_ids)
        end_word_repr = self.get_word_embedding(end_ids)
        state_repr = (word_repr - end_word_repr).mean(dim='word')
        return state_repr

    def get_policy(self, state_or_ids: Union[VocabState, LT]) -> Distribution:
        """Get policy distribution based on current state (and end state)."""
        if isinstance(state_or_ids, VocabState):
            ids = state_or_ids.ids
        else:
            ids = state_or_ids
        state_repr = self.get_state_repr(ids, self.end_state.ids)

        action_logits = self.action_predictor(state_repr)

        with NoName(action_logits):
            policy = torch.distributions.Categorical(logits=action_logits)
        return policy

    def sample_action(self, policy: Distribution) -> SoundChangeAction:
        action_id = policy.sample().item()
        return self.action_space.get_action(action_id)

    def _get_rewards(self, agent_inputs: AgentInputs) -> FT:
        """Obtain rewards that will be multiplied with log_probs."""
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

    def forward(self, agent_inputs: AgentInputs) -> Tuple[FT, FT]:
        with ScopedCache('word_embedding'):
            policy = self.get_policy(agent_inputs.id_seqs)
            with NoName(agent_inputs.action_ids):
                log_probs = policy.log_prob(agent_inputs.action_ids)
            # Compute rewards to go.
            rews = self._get_rewards(agent_inputs)
        return log_probs, rews


class A2C(VanillaPolicyGradient):

    def __init__(self, char_emb: CharEmbedding, action_space: SoundChangeActionSpace, end_state: VocabState):
        super().__init__(char_emb, action_space, end_state)
        self.value_predictor = nn.Sequential(
            nn.Linear(char_emb.embedding_dim, 1),
            nn.Flatten(-2, -1))

    def get_values(self, curr_ids: LT, end_ids: LT) -> FT:
        state_repr = self.get_state_repr(curr_ids, end_ids)
        with NoName(state_repr):
            values = self.value_predictor(state_repr)
        return values

    def _get_rewards(self, agent_inputs: AgentInputs) -> Tuple[FT, FT, FT]:
        end_ids = self.end_state.ids
        values = self.get_values(agent_inputs.id_seqs, end_ids)
        next_values = self.get_values(agent_inputs.next_id_seqs, end_ids)
        expected_rews = agent_inputs.rewards + next_values
        advantages = expected_rews - values
        rtgs = super()._get_rewards(agent_inputs)
        try:
            self._cnt += 1
        except:
            self._cnt = 1
        if self._cnt == 300:
            breakpoint()  # BREAKPOINT(j_luo)
        # return advantages, values, expected_rews
        return rtgs, values, expected_rews

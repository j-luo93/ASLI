from dataclasses import dataclass, field
from typing import List, Union

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from dev_misc import FT, LT, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics, init_params
from sound_law.model.module import CharEmbedding, EmbParams, PhonoEmbedding

from .action import SoundChangeAction, SoundChangeActionSpace
from .trajectory import Trajectory, VocabState


@dataclass
class AgentInputs:
    trajectories: List[Trajectory]
    id_seqs: LT
    action_ids: LT
    rewards: FT

    @property
    def batch_size(self) -> int:
        return self.action_ids.size('batch')


class ActorCritic(nn.Module):

    def __init__(self, char_emb: CharEmbedding, action_space: SoundChangeActionSpace, end_state: VocabState):
        super().__init__()
        self.char_emb = char_emb
        self.action_space = action_space
        num_actions = len(action_space)
        self.action_predictor = nn.Linear(char_emb.embedding_dim, num_actions)
        self.end_state = end_state

    def get_policy(self, state_or_ids: Union[VocabState, LT]) -> Distribution:
        if isinstance(state_or_ids, VocabState):
            ids = state_or_ids.ids
        else:
            ids = state_or_ids
        # HACK(j_luo) names are messed up.
        names = ('pos', 'word', 'emb')
        # HACK(j_luo) shape?
        if len(ids.shape) == 3:
            names = ('batch', ) + names
        emb = self.char_emb(ids).rename(*names)
        word_repr = emb.mean(dim='pos')

        end_emb = self.char_emb(self.end_state.ids).rename('pos', 'word', 'emb')
        end_word_repr = end_emb.mean(dim='pos')

        state_repr = (word_repr - end_word_repr).mean(dim='word')

        action_logits = self.action_predictor(state_repr)

        with NoName(action_logits):
            policy = torch.distributions.Categorical(logits=action_logits)
        return policy

    def sample_action(self, policy: Distribution) -> SoundChangeAction:
        action_id = policy.sample().item()
        return self.action_space.get_action(action_id)

    def forward(self, agent_inputs: AgentInputs) -> Metrics:
        policy = self.get_policy(agent_inputs.id_seqs)
        with NoName(agent_inputs.action_ids):
            log_probs = policy.log_prob(agent_inputs.action_ids)

        # Compute rewards to go.
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

        losses = (-log_probs * rtgs)
        loss = Metric('loss', losses.sum(), agent_inputs.batch_size)
        return Metrics(loss)

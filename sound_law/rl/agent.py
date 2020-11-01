from __future__ import annotations

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from dev_misc import BT, FT, LT, NDA, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics, init_params
from dev_misc.utils import ScopedCache, cacheable
from sound_law.model.base_model import get_emb_params
from sound_law.model.module import (CharEmbedding, EmbParams, PhonoEmbedding,
                                    get_embedding)

from .action import SoundChangeAction, SoundChangeActionSpace
from .reward import get_rtgs_list
from .trajectory import Trajectory, VocabState


@dataclass
class AgentInputs:
    trajectories: List[Trajectory]
    id_seqs: LT
    next_id_seqs: LT
    action_ids: LT
    rewards: FT
    done: BT
    action_masks: BT

    @property
    def batch_size(self) -> int:
        return self.action_ids.size('batch')


@dataclass
class RewardOutputs:
    """This stores all relevant outputs related to rewards, including advantages and policy evaluations."""
    rtgs: Optional[FT] = None  # rewards-to-go
    expected: Optional[FT] = None
    values: Optional[FT] = None
    advantages: Optional[FT] = None


@dataclass
class AgentOutputs:
    log_probs: Optional[FT] = None
    entropy: Optional[FT] = None
    rew_outputs: Optional[RewardOutputs] = None


# @cacheable(switch='word_embedding')
def _get_word_embedding(char_emb: PhonoEmbedding, ids: LT, cnn: nn.Module = None) -> FT:
    """Get word embeddings based on ids."""
    names = ids.names + ('emb',)
    emb = char_emb(ids).rename(*names)
    if cnn is not None:
        if emb.ndim == 4:
            emb = emb.align_to('batch', 'word', 'emb', 'pos')
            bs, ws, hs, l = emb.shape
            ret = cnn(emb.rename(None).reshape(bs * ws, hs, l)).view(bs, ws, hs, -1).max(dim=-1)[0]
            return ret.rename('batch', 'word', 'emb')
        else:
            emb = emb.align_to('word', 'emb', 'pos')
            ret = cnn(emb.rename(None)).max(dim=-1)[0]
            return ret.rename('word', 'emb')

    return emb.mean(dim='pos')


def _get_state_repr(char_emb: PhonoEmbedding, curr_ids: LT, end_ids: LT, cnn: nn.Module = None) -> FT:
    """Get state representation used for action prediction."""
    word_repr = _get_word_embedding(char_emb, curr_ids, cnn=cnn)
    end_word_repr = _get_word_embedding(char_emb, end_ids, cnn=cnn)
    state_repr = (word_repr - end_word_repr).mean(dim='word')
    return state_repr


def _get_rewards_to_go(agent_inputs: AgentInputs) -> FT:
    rewards = [tr.rewards for tr in agent_inputs.trajectories]
    rtgs = get_rtgs_list(rewards, g.discount)
    return get_tensor(np.concatenate(rtgs))


@dataclass
class Cnn1dParams:
    input_size: int
    hidden_size: int
    kernel_size: int
    num_layers: int


def get_cnn1d(cnn1d_params: Cnn1dParams) -> nn.Module:
    layers = list()
    for i in range(cnn1d_params.num_layers):
        layers.append(nn.Conv1d(cnn1d_params.input_size,
                                cnn1d_params.hidden_size,
                                cnn1d_params.kernel_size))
        if i != cnn1d_params.num_layers - 1:
            layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):

    def __init__(self, char_emb: CharEmbedding,
                 cnn: nn.Module,
                 proj: nn.Module,
                 action_space: SoundChangeActionSpace):
        super().__init__()
        self.char_emb = char_emb
        self.cnn = cnn
        self.proj = proj
        self.action_space = action_space

    @classmethod
    def from_params(cls, emb_params: EmbParams,
                    cnn1d_params: Cnn1dParams,
                    action_space: SoundChangeActionSpace) -> PolicyNetwork:
        char_emb = get_embedding(emb_params)
        cnn = get_cnn1d(cnn1d_params)
        input_size = cnn1d_params.hidden_size
        num_actions = len(action_space)
        proj = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, num_actions))
        return cls(char_emb, cnn, proj, action_space)

    def forward(self,
                curr_ids: LT,
                end_ids: LT,
                action_masks: BT) -> Distribution:
        """Get policy distribution based on current state (and end state). If ids are passed, we have to specify action masks directly."""
        state_repr = _get_state_repr(self.char_emb, curr_ids, end_ids, cnn=self.cnn)

        action_logits = self.proj(state_repr)
        action_logits = torch.where(action_masks, action_logits,
                                    torch.full_like(action_logits, -999.9))

        with NoName(action_logits):
            policy = torch.distributions.Categorical(logits=action_logits)
        return policy


class ValueNetwork(nn.Module):

    def __init__(self, char_emb: CharEmbedding, cnn: nn.Module, regressor: nn.Module):
        super().__init__()
        self.char_emb = char_emb
        self.cnn = cnn
        self.regressor = regressor

    @classmethod
    def from_params(cls,
                    emb_params: EmbParams,
                    cnn1d_params: Cnn1dParams,
                    char_emb: Optional[CharEmbedding] = None,
                    cnn: Optional[nn.Module] = None) -> ValueNetwork:
        char_emb = char_emb or get_embedding(emb_params)
        cnn = cnn or get_cnn1d(cnn1d_params)
        input_size = cnn1d_params.hidden_size
        regressor = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, 1),
            nn.Flatten(-2, -1))
        return ValueNetwork(char_emb, cnn, regressor)

    def forward(self, curr_ids: LT, end_ids: LT, done: Optional[BT] = None) -> FT:
        """Get policy evaluation. if `done` is provided, we get values for s1 instead of s0. In that case, end states should have values set to 0."""
        state_repr = _get_state_repr(self.char_emb, curr_ids, end_ids, cnn=self.cnn)
        with NoName(state_repr):
            values = self.regressor(state_repr)
        if done is not None:
            values = torch.where(done, torch.zeros_like(values), values)
        return values


def get_bool_context(attr_name: str):

    @contextmanager
    def method(self, value: bool):
        old_value = getattr(self, attr_name)
        setattr(self, attr_name, value)
        yield
        setattr(self, attr_name, old_value)

    return method


class BasePG(nn.Module, metaclass=ABCMeta):

    add_argument('discount', dtype=float, default=1.0, msg='Discount for computing rewards.')

    def __init__(self, num_chars: int,
                 action_space: SoundChangeActionSpace,
                 end_state: VocabState,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):
        super().__init__()
        emb_params = get_emb_params(num_chars, phono_feat_mat, special_ids)
        cnn1d_params = Cnn1dParams(g.char_emb_size, g.hidden_size, 3, g.num_layers)
        self.policy_net = PolicyNetwork.from_params(emb_params, cnn1d_params, action_space)
        self.value_net = self._get_value_net(emb_params, cnn1d_params)
        self.end_state = end_state
        self._policy_grad = True
        self._value_grad = True

    policy_grad = get_bool_context('_policy_grad')
    value_grad = get_bool_context('_value_grad')

    @abstractmethod
    def _get_value_net(self, emb_params: EmbParams, cnn1d_params: Cnn1dParams) -> Optional[ValueNetwork]: ...

    def get_values(self, curr_ids: LT, end_ids: LT, done: Optional[BT] = None) -> FT:
        if self.value_net is None:
            raise TypeError(f'There is no value net.')
        with torch.set_grad_enabled(self._value_grad):
            return self.value_net(curr_ids, end_ids, done=done)

    def get_policy(self, state_or_ids: Union[VocabState, LT], action_masks: BT) -> Distribution:
        """Get policy distribution based on current state (and end state). If ids are passed, we have to specify action masks directly."""
        if isinstance(state_or_ids, VocabState):
            curr_ids = state_or_ids.ids
        else:
            curr_ids = state_or_ids
        end_ids = self.end_state.ids
        with torch.set_grad_enabled(self._policy_grad):
            return self.policy_net(curr_ids, end_ids, action_masks)

    def sample_action(self, policy: Distribution) -> SoundChangeAction:
        with torch.set_grad_enabled(self._policy_grad):
            action_id = policy.sample().item()
            return self.policy_net.action_space.get_action(action_id)

    def forward(self, agent_inputs: AgentInputs,
                ret_log_probs: bool = True,
                ret_entropy: bool = True,
                ret_rewards: bool = True) -> AgentOutputs:
        """Obtain agent outputs by feeding agent inputs. Use `ret_log_probs`, `ret_entropy` and `re_rewards`
        to specify which outputs to return.
        """
        log_probs = entropy = rew_outputs = None
        # with ScopedCache('word_embedding'):
        if ret_log_probs or ret_entropy:
            policy = self.get_policy(agent_inputs.id_seqs, agent_inputs.action_masks)
            if ret_entropy:
                entropy = policy.entropy()
        if ret_log_probs:
            with NoName(agent_inputs.action_ids), torch.set_grad_enabled(self._policy_grad):
                log_probs = policy.log_prob(agent_inputs.action_ids)
        if ret_rewards:
            rew_outputs = self._get_reward_outputs(agent_inputs)
        return AgentOutputs(log_probs, entropy, rew_outputs)

    @abstractmethod
    def _get_reward_outputs(self, agent_inputs: AgentInputs) -> RewardOutputs: ...


class VanillaPolicyGradient(BasePG):
    """Simplest pg agent without value net."""

    def _get_value_net(self, emb_params: EmbParams, cnn1d_params: Cnn1dParams):
        return None

    def _get_reward_outputs(self, agent_inputs: AgentInputs) -> RewardOutputs:
        """Obtain outputs related to reward."""
        rtgs = _get_rewards_to_go(agent_inputs)
        return RewardOutputs(rtgs)


class A2C(BasePG):

    add_argument('critic_target', dtype=str, default='expected',
                 choices=['rtg', 'expected'], msg='What is the target for value net.')
    add_argument('critic_mode', dtype=str, default='ac',
                 choices=['ac', 'mc'], msg='How to use the critic.')
    # add_argument('gae_lambda', dtype=float, default=0.95, msg='Lambda value for GAE.')

    def _get_value_net(self, emb_params, cnn1d_params):
        char_emb = cnn = None
        if not g.separate_value:
            char_emb = self.policy_net.char_emb
            cnn = self.policy_net.cnn
        return ValueNetwork.from_params(emb_params, cnn1d_params, char_emb=char_emb, cnn=cnn)

    def _get_reward_outputs(self, agent_inputs: AgentInputs) -> RewardOutputs:
        end_ids = self.end_state.ids
        values = self.get_values(agent_inputs.id_seqs, end_ids)
        rtgs = expected = None
        if g.critic_target == 'rtg' or g.critic_mode == 'mc':
            rtgs = _get_rewards_to_go(agent_inputs)
        if g.critic_target == 'expected' or g.critic_mode == 'ac':
            next_values = self.get_values(agent_inputs.next_id_seqs, end_ids, agent_inputs.done)
            # This computes the expected rewards-to-go.
            expected = (agent_inputs.rewards + g.discount * next_values).detach()

        if g.critic_mode == 'mc':
            advantages = rtgs - values.detach()
        else:
            advantages = expected - values.detach()

        return RewardOutputs(rtgs=rtgs, expected=expected, values=values, advantages=advantages)

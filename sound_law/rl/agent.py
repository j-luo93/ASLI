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
from .mcts_fast import (  # pylint: disable=no-name-in-module
    parallel_get_action_masks, parallel_get_sparse_action_masks,
    parallel_stack_ids)
from .reward import get_rtgs_dense  # pylint: disable=no-name-in-module
from .reward import get_rtgs_list
from .trajectory import Trajectory, VocabState


@dataclass
class AgentInputs:
    trajectories: List[Trajectory]
    id_seqs: LT
    rewards: FT
    action_masks: BT
    next_id_seqs: Optional[LT] = None
    action_ids: Optional[LT] = None
    done: Optional[BT] = None
    indices: Optional[LT] = None
    steps: Optional[LT] = None

    @property
    def batch_size(self) -> int:
        return self.action_ids.size('batch')

    @property
    def offsets(self) -> NDA:
        return np.cumsum(np.asarray([len(tr) for tr in self.trajectories]), dtype='long')

    @classmethod
    def from_trajectories(cls,
                          trs: List[Trajectory],
                          action_space: SoundChangeActionSpace,
                          sparse: bool = False) -> AgentInputs:
        def gather(attr: str):
            """Gather information from trajectory edges. See definition for `TrEdge`."""
            ret = list()
            for tr in trs:
                ret.extend([getattr(edge, attr) for edge in tr])
            return ret

        states = gather('s0')
        id_seqs = get_tensor(parallel_stack_ids(states, g.num_workers)).rename('batch', 'pos', 'word')
        rewards = get_tensor(gather('r')).rename('batch')
        if sparse:
            indices, action_masks, _ = parallel_get_sparse_action_masks(states, g.num_workers)
            indices = get_tensor(indices).rename('batch', 'action')
        else:
            action_masks = parallel_get_action_masks(states, action_space, g.num_workers)
            indices = None
        action_masks = get_tensor(action_masks).rename('batch', 'action')
        steps = next_id_seqs = action_ids = done = None
        if not g.use_mcts:
            next_id_seqs = parallel_stack_ids(gather('s1'), num_threads=g.num_workers)
            next_id_seqs = get_tensor(next_id_seqs).rename('batch', 'pos', 'word')
            action_ids = get_tensor([a.action_id for a in gather('a')]).rename('batch')
            done = get_tensor(gather('done')).rename('batch')
        if g.use_finite_horizon:
            steps = list()
            for tr in trs:
                steps.extend(list(range(len(tr))))
            steps = get_tensor(steps)
        return AgentInputs(trs, id_seqs, rewards, action_masks,
                           next_id_seqs=next_id_seqs,
                           action_ids=action_ids,
                           done=done,
                           indices=indices,
                           steps=steps)


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


@cacheable(switch='word_embedding')
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


class FactorizedProjection(nn.Module):
    """A factorized projection layer that predicts the before ids and the after ids."""

    def __init__(self, input_size: int, action_space: SoundChangeActionSpace):
        super().__init__()
        num_ids = len(action_space.abc)
        self.before_potential = nn.Linear(input_size, num_ids)
        self.after_potential = nn.Linear(input_size, num_ids)
        if g.use_conditional:
            self.pre_potential = nn.Linear(input_size, num_ids)
        self.action_space = action_space

    def forward(self, inp: FT, sparse: bool = False, indices: Optional[LT] = None) -> FT:
        is_2d = inp.ndim == 2
        if g.use_conditional and not is_2d:
            raise RuntimeError(f'Not sure why you end up here.')

        def get_potential(attr: str):
            a2i = getattr(self.action_space, f'action2{attr}')
            mod = getattr(self, f'{attr}_potential')
            potential = mod(inp)
            with NoName(potential, indices):
                if sparse:
                    a2i = a2i[indices]
                    # NOTE(j_luo) For conditional rules, mask out those that are not.
                    if attr == 'pre':
                        pre_mask = a2i == -1
                        a2i = torch.where(pre_mask, torch.zeros_like(a2i), a2i)
                        ret = potential.gather(1, a2i)
                        ret = torch.where(pre_mask, torch.zeros_like(ret), ret)
                        return ret
                    return potential.gather(1, a2i)
                elif is_2d:
                    return potential[:, a2i]
                else:
                    return potential[a2i]

        bp = get_potential('before')
        ap = get_potential('after')
        if g.use_conditional:
            pp = get_potential('pre')
            ret = bp + ap + pp
        else:
            ret = bp + ap
        names = ('batch', ) * is_2d + ('action',)
        return ret.rename(*names)


class SparseProjection(nn.Module):
    """A projection layer that can be selectively computed on given indices."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.weight = nn.Parameter(nn.init.xavier_uniform(torch.randn(num_classes, input_size)))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, inp: FT, sparse: bool = False, indices: Optional[LT] = None):
        with NoName(inp):
            if sparse:
                w = self.weight[indices]
                b = self.bias[indices]
                out = torch.bmm(w, inp.unsqueeze(dim=-1)).squeeze(dim=-1) + b
                return out
            else:
                return torch.addmm(self.bias, inp, self.weight.t())


class PolicyNetwork(nn.Module):

    def __init__(self, char_emb: CharEmbedding,
                 cnn: nn.Module,
                 hidden: nn.Module,
                 proj: nn.Module,
                 action_space: SoundChangeActionSpace):
        super().__init__()
        self.char_emb = char_emb
        self.cnn = cnn
        self.hidden = hidden
        self.proj = proj
        self.action_space = action_space

    @ classmethod
    def from_params(cls, emb_params: EmbParams,
                    cnn1d_params: Cnn1dParams,
                    action_space: SoundChangeActionSpace) -> PolicyNetwork:
        char_emb = get_embedding(emb_params)
        cnn = get_cnn1d(cnn1d_params)
        input_size = cnn1d_params.hidden_size
        num_actions = len(action_space)
        hidden = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh())
        if g.factorize_actions:
            proj = FactorizedProjection(input_size // 2, action_space)
        else:
            proj = SparseProjection(input_size // 2, num_actions)
        return cls(char_emb, cnn, hidden, proj, action_space)

    def forward(self,
                curr_ids: LT,
                end_ids: LT,
                action_masks: BT,
                sparse: bool = False,
                indices: Optional[LT] = None) -> Distribution:
        """Get policy distribution based on current state (and end state)."""
        if sparse and indices is None:
            raise TypeError(f'Must provide `indices` in sparse mode.')

        state_repr = _get_state_repr(self.char_emb, curr_ids, end_ids, cnn=self.cnn)

        hid = self.hidden(state_repr)
        if sparse:
            action_logits = self.proj(hid, indices=indices, sparse=True)
        else:
            action_logits = self.proj(hid, sparse=False)
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
            nn.Linear(input_size + g.use_finite_horizon, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, 1))
        return ValueNetwork(char_emb, cnn, regressor)

    def forward(self, curr_ids: LT, end_ids: LT, steps: Optional[LT] = None, done: Optional[BT] = None) -> FT:
        """Get policy evaluation. if `done` is provided, we get values for s1 instead of s0.
        In that case, end states should have values set to 0.
        `step` should start with 0.
        """
        # In finite mode, if this is the last step, and we are evaluating s1, we should return 0 value.
        state_repr = _get_state_repr(self.char_emb, curr_ids, end_ids, cnn=self.cnn)
        # NOTE(j_luo) If s1 is being evaluated, we should increment `step`.
        if done is not None and g.use_finite_horizon:
            steps = steps + 1
        with NoName(state_repr, steps):
            if g.use_finite_horizon:
                rel_step = steps.float() / g.max_rollout_length
                state_repr = torch.cat([state_repr, rel_step.unsqueeze(dim=-1)], dim=-1)
            values = self.regressor(state_repr).squeeze(dim=-1)
        # Deal with special cases. We start with final step case, and then overwrite it if done.
        if g.use_finite_horizon:
            final_step = steps == g.max_rollout_length
            values = torch.where(final_step, torch.zeros_like(values), values)
        if done is not None:
            # NOTE(j_luo) Use final reward for the value of the end state.
            values = torch.where(done, torch.full_like(values, g.final_reward), values)
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
    add_argument('use_finite_horizon', dtype=bool, default=False, msg='Flag to use finite horizon.')

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

    def get_values(self,
                   curr_state_or_ids: Union[VocabState, LT],
                   steps: Optional[Union[int, LT]] = None,
                   done: Optional[BT] = None) -> FT:
        if self.value_net is None:
            raise TypeError(f'There is no value net.')
        if g.use_finite_horizon and steps is None:
            raise TypeError(f'Must pass the step if finite horizon is used.')
        if isinstance(curr_state_or_ids, VocabState):
            curr_ids = curr_state_or_ids.tensor
        else:
            curr_ids = curr_state_or_ids
        end_ids = self.end_state.tensor

        if isinstance(steps, int):
            steps = torch.full([curr_ids.shape[0]], steps, dtype=torch.long, device=end_ids.device)
        with torch.set_grad_enabled(self._value_grad):
            return self.value_net(curr_ids, end_ids, steps=steps, done=done)

    def get_policy(self,
                   state_or_ids: Union[VocabState, LT],
                   action_masks: BT,
                   sparse: bool = False,
                   indices: Optional[LT] = None) -> Distribution:
        """Get policy distribution based on current state (and end state)."""
        if isinstance(state_or_ids, VocabState):
            curr_ids = state_or_ids.tensor
        else:
            curr_ids = state_or_ids
        end_ids = self.end_state.tensor
        with torch.set_grad_enabled(self._policy_grad):
            return self.policy_net(curr_ids, end_ids,
                                   action_masks,
                                   sparse=sparse,
                                   indices=indices)

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
        with ScopedCache('word_embedding'):
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
    add_argument('use_gae', dtype=bool, default=False, msg='Flag to use GAE.')
    add_argument('gae_lambda', dtype=float, default=1.0, msg='Lambda value for GAE.')

    def _get_value_net(self, emb_params, cnn1d_params):
        char_emb = cnn = None
        if not g.separate_value:
            char_emb = self.policy_net.char_emb
            cnn = self.policy_net.cnn
        return ValueNetwork.from_params(emb_params, cnn1d_params, char_emb=char_emb, cnn=cnn)

    def _get_reward_outputs(self, agent_inputs: AgentInputs) -> RewardOutputs:
        values = self.get_values(agent_inputs.id_seqs)
        rtgs = expected = None
        if g.critic_target == 'rtg' or g.critic_mode == 'mc':
            rtgs = _get_rewards_to_go(agent_inputs)
        if g.critic_target == 'expected' or g.critic_mode == 'ac':
            next_values = self.get_values(agent_inputs.next_id_seqs, done=agent_inputs.done)
            # This computes the expected rewards-to-go.
            expected = (agent_inputs.rewards + g.discount * next_values).detach()

        if g.critic_mode == 'mc':
            advantages = rtgs - values.detach()
        elif g.use_gae:
            deltas = (expected - values.detach()).cpu().numpy()
            offsets = agent_inputs.offsets
            advantages = get_tensor(get_rtgs_dense(deltas, offsets, g.discount * g.gae_lambda))
        else:
            advantages = expected - values.detach()

        return RewardOutputs(rtgs=rtgs, expected=expected, values=values, advantages=advantages)

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from dev_misc import BT, FT, LT, NDA, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics, init_params
from dev_misc.utils import ScopedCache, cacheable
from sound_law.s2s.base_model import get_emb_params
from sound_law.s2s.module import (CharEmbedding, EmbParams, PhonoEmbedding,
                                  get_embedding)

from .action import SoundChangeAction, SoundChangeActionSpace
from .mcts_cpp import (  # pylint: disable=no-name-in-module
    parallel_get_sparse_action_masks, parallel_stack_ids)
from .module import Cnn1dParams, PolicyNetwork, ValueNetwork
from .reward import get_rtgs_dense, get_rtgs_list  # pylint: disable=no-name-in-module
from .trajectory import Trajectory, TrEdge, VocabState


@dataclass
class AgentInputs:
    edges: List[TrEdge]
    id_seqs: LT
    rewards: FT
    action_masks: BT
    next_id_seqs: Optional[LT] = None
    action_ids: Optional[LT] = None
    done: Optional[BT] = None
    indices: Optional[NDA] = None
    steps: Optional[LT] = None
    trajectories: Optional[List[Trajectory]] = None
    rtgs: Optional[FT] = None

    def __post_init__(self):
        if self.trajectories is None and self.edges is None:
            raise TypeError(f'You must have either trajectories or edges.')

    @property
    def batch_size(self) -> int:
        return self.action_ids.size('batch')

    @property
    def offsets(self) -> NDA:
        return np.cumsum(np.asarray([len(tr) for tr in self.trajectories]), dtype='long')

    @classmethod
    def from_edges(cls,
                   edges: List[TrEdge],
                   action_space: SoundChangeActionSpace,
                   sparse: bool = False) -> AgentInputs:

        def gather(attr: str):
            """Gather information from trajectory edges. See definition for `TrEdge`."""
            return [getattr(edge, attr) for edge in edges]

        states = gather('s0')
        id_seqs = get_tensor(parallel_stack_ids(states, g.num_workers)).rename('batch', 'pos', 'word')
        rewards = get_tensor(gather('r')).rename('batch')
        if sparse:
            indices, action_masks, _ = parallel_get_sparse_action_masks(states, g.num_workers)
            # indices = get_tensor(indices).rename('batch', 'action')
        else:
            action_masks = parallel_get_action_masks(states, action_space, g.num_workers)
            indices = None
        action_masks = get_tensor(action_masks).rename('batch', 'action')
        rtgs = steps = next_id_seqs = action_ids = done = None
        if not g.use_mcts:
            next_id_seqs = parallel_stack_ids(gather('s1'), num_threads=g.num_workers)
            next_id_seqs = get_tensor(next_id_seqs).rename('batch', 'pos', 'word')
            action_ids = get_tensor([a.action_id for a in gather('a')]).rename('batch')
            done = get_tensor(gather('done')).rename('batch')
        if g.use_finite_horizon:
            steps = get_tensor(gather('step'))
        if edges[0].rtg is not None:
            rtgs = get_tensor(gather('rtg')).rename('batch')
        return cls(edges, id_seqs, rewards, action_masks,
                   next_id_seqs=next_id_seqs,
                   action_ids=action_ids,
                   done=done,
                   indices=indices,
                   steps=steps,
                   rtgs=rtgs)

    @classmethod
    def from_trajectories(cls,
                          trs: List[Trajectory],
                          action_space: SoundChangeActionSpace,
                          sparse: bool = False) -> AgentInputs:
        edges: List[TrEdge] = list()
        for tr in trs:
            edges.extend(list(tr))

        ret = cls.from_edges(edges, action_space, sparse=sparse)
        ret.trajectories = trs
        return ret


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


def _get_rewards_to_go(agent_inputs: AgentInputs) -> FT:
    rewards = [tr.rewards for tr in agent_inputs.trajectories]
    rtgs = get_rtgs_list(rewards, g.discount)
    return get_tensor(np.concatenate(rtgs))


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
        cnn1d_params = Cnn1dParams(g.char_emb_size, g.hidden_size, 3, g.num_layers, g.dropout)
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
                   indices: Optional[NDA] = None) -> Distribution:
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
        with ScopedCache('state_repr'):
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
        enc = None
        if not g.separate_value:
            enc = self.policy_net.enc
        return ValueNetwork.from_params(emb_params, cnn1d_params,
                                        enc=enc)

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

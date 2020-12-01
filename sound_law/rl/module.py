from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from dev_misc import BT, FT, LT, NDA, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName
from dev_misc.utils import ScopedCache, cacheable
from sound_law.s2s.module import (CharEmbedding, EmbParams, PhonoEmbedding,
                                  get_embedding)

from .action import SoundChangeAction, SoundChangeActionSpace
from .mcts_fast import Code, parallel_gather_action_info  # pylint: disable=no-name-in-module


class FactorizedProjection(nn.Module):
    """A factorized projection layer that predicts the before ids and the after ids."""

    def __init__(self, input_size: int, action_space: SoundChangeActionSpace):
        super().__init__()
        num_ids = len(action_space.abc)
        self.before_potential = nn.Linear(input_size, num_ids)
        self.after_potential = nn.Linear(input_size, num_ids)
        if g.use_conditional:
            self.pre_potential = nn.Linear(input_size, num_ids)
            self.post_potential = nn.Linear(input_size, num_ids)
            self.d_pre_potential = nn.Linear(input_size, num_ids)
            self.d_post_potential = nn.Linear(input_size, num_ids)
        self.action_space = action_space

    def forward(self, inp: FT, sparse: bool = False, indices: Optional[NDA] = None) -> FT:
        is_2d = inp.ndim == 2
        if g.use_conditional and not is_2d:
            raise RuntimeError(f'Not sure why you end up here.')

        assert sparse, 'Cannot deal with dense action space.'

        def get_potential(attr: str, code: int):
            mod = getattr(self, f'{attr}_potential')
            potential = mod(inp)
            with NoName(potential):
                if sparse:
                    # a2i = (f'{attr}_id', indices)
                    # a2i = a2i[indices]
                    a2i = get_tensor(parallel_gather_action_info(self.action_space, indices, code, g.num_workers))
                    # NOTE(j_luo) For conditional rules, mask out those that are not.
                    if attr in ['pre', 'd_pre', 'post', 'd_post']:
                        mask = a2i == -1
                        a2i = torch.where(mask, torch.zeros_like(a2i), a2i)
                        ret = potential.gather(1, a2i)
                        ret = torch.where(mask, torch.zeros_like(ret), ret)
                        return ret
                    return potential.gather(1, a2i)
                elif is_2d:
                    return potential[:, a2i]
                else:
                    return potential[a2i]

        bp = get_potential('before', Code.BEFORE)
        ap = get_potential('after', Code.AFTER)
        if g.use_conditional:
            prep = get_potential('pre', Code.PRE)
            d_prep = get_potential('d_pre', Code.D_PRE)
            postp = get_potential('post', Code.POST)
            d_postp = get_potential('d_post', Code.D_POST)
            ret = bp + ap + prep + d_prep + postp + d_postp
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


@dataclass
class Cnn1dParams:
    input_size: int
    hidden_size: int
    kernel_size: int
    num_layers: int
    dropout: float


def get_cnn1d(cnn1d_params: Cnn1dParams) -> nn.Module:
    layers = list()
    for i in range(cnn1d_params.num_layers):
        layers.append(nn.Conv1d(cnn1d_params.input_size,
                                cnn1d_params.hidden_size,
                                cnn1d_params.kernel_size))
        if i != cnn1d_params.num_layers - 1:
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(cnn1d_params.dropout))
    layers.append(nn.Dropout(cnn1d_params.dropout))
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """Encode a vocab state."""

    def __init__(self, char_emb: CharEmbedding, cnn: nn.Module):
        super().__init__()
        self.char_emb = char_emb
        self.cnn = cnn

    @classmethod
    def from_params(cls, emb_params: EmbParams, cnn1d_params: Cnn1dParams):
        char_emb = get_embedding(emb_params)
        cnn = get_cnn1d(cnn1d_params)
        return cls(char_emb, cnn)

    # FIXME(j_luo) this might not be instance-specific.
    @cacheable(switch='state_repr')
    def forward(self, curr_ids: LT, end_ids: LT):
        word_repr = self._get_word_embedding(curr_ids)
        end_word_repr = self._get_word_embedding(end_ids)
        state_repr = (word_repr - end_word_repr).mean(dim='word')
        return state_repr

    def _get_word_embedding(self, ids: LT) -> FT:
        """Get word embeddings based on ids."""
        names = ids.names + ('emb',)
        emb = self.char_emb(ids).rename(*names)
        if emb.ndim == 4:
            emb = emb.align_to('batch', 'word', 'emb', 'pos')
            bs, ws, es, l = emb.shape
            # NOTE(j_luo) embedding size might not match hidden size.
            emb_3d = emb.rename(None).reshape(bs * ws, es, -1)
            ret = self.cnn(emb_3d).view(bs, ws, g.hidden_size, -1).max(dim=-1)[0]
            return ret.rename('batch', 'word', 'emb')
        else:
            emb = emb.align_to('word', 'emb', 'pos')
            ret = self.cnn(emb.rename(None)).max(dim=-1)[0]
            return ret.rename('word', 'emb')

        return emb.mean(dim='pos')


class PolicyNetwork(nn.Module):

    def __init__(self,
                 enc: StateEncoder,
                 hidden: nn.Module,
                 proj: nn.Module,
                 action_space: SoundChangeActionSpace):
        super().__init__()
        self.enc = enc
        self.hidden = hidden
        self.proj = proj
        self.action_space = action_space

    @classmethod
    def from_params(cls, emb_params: EmbParams,
                    cnn1d_params: Cnn1dParams,
                    action_space: SoundChangeActionSpace) -> PolicyNetwork:
        enc = StateEncoder.from_params(emb_params, cnn1d_params)
        input_size = cnn1d_params.hidden_size
        num_actions = len(action_space)
        hidden = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Dropout(g.dropout))
        if g.factorize_actions:
            proj = FactorizedProjection(input_size // 2, action_space)
        else:
            proj = SparseProjection(input_size // 2, num_actions)
        return cls(enc, hidden, proj, action_space)

    def forward(self,
                curr_ids: LT,
                end_ids: LT,
                action_masks: BT,
                sparse: bool = False,
                indices: Optional[NDA] = None) -> Distribution:
        """Get policy distribution based on current state (and end state)."""
        if sparse and indices is None:
            raise TypeError(f'Must provide `indices` in sparse mode.')

        state_repr = self.enc(curr_ids, end_ids)

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

    def __init__(self, enc: StateEncoder, regressor: nn.Module):
        super().__init__()
        self.enc = enc
        self.regressor = regressor

    @classmethod
    def from_params(cls,
                    emb_params: EmbParams,
                    cnn1d_params: Cnn1dParams,
                    enc: Optional[StateEncoder] = None) -> ValueNetwork:
        enc = enc or StateEncoder.from_params(emb_params, cnn1d_params)
        input_size = cnn1d_params.hidden_size
        regressor = nn.Sequential(
            nn.Linear(input_size + g.use_finite_horizon, input_size // 2),
            nn.Tanh(),
            nn.Dropout(g.dropout),
            nn.Linear(input_size // 2, 1))
        return ValueNetwork(enc, regressor)

    def forward(self, curr_ids: LT, end_ids: LT, steps: Optional[LT] = None, done: Optional[BT] = None) -> FT:
        """Get policy evaluation. if `done` is provided, we get values for s1 instead of s0.
        In that case, end states should have values set to 0.
        `step` should start with 0.
        """
        state_repr = self.enc(curr_ids, end_ids)
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

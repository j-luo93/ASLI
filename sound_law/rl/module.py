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

    @classmethod
    def from_params(cls, emb_params: EmbParams,
                    cnn1d_params: Cnn1dParams,
                    action_space: SoundChangeActionSpace) -> PolicyNetwork:
        char_emb = get_embedding(emb_params)
        cnn = get_cnn1d(cnn1d_params)
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
            nn.Dropout(g.dropout),
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

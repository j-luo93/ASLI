from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.functional import normalize

from dev_misc import BT, FT, LT, get_zeros
from dev_misc.devlib.named_tensor import NameHelper, NoName
from sound_law.model.lstm_state import LstmStatesByLayers, LstmStateTuple

LstmOutputsByLayers = Tuple[FT, LstmStatesByLayers]


@dataclass
class LstmParams:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool


class MultiLayerLSTMCell(nn.Module):
    """An LSTM cell with multiple layers."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)

        cells = [nn.LSTMCell(input_size, hidden_size)]
        for _ in range(self.num_layers - 1):
            cells.append(nn.LSTMCell(hidden_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    # IDEA(j_luo) write a function to generate simple `from_params` classmethods?
    @classmethod
    def from_params(cls, lstm_params: LstmParams) -> MultiLayerLSTMCell:
        return cls(lstm_params.input_size,
                   lstm_params.hidden_size,
                   lstm_params.num_layers,
                   lstm_params.dropout)

    def forward(self, input_: FT, state: LstmStatesByLayers, state_direction: Optional[str] = None) -> LstmOutputsByLayers:
        assert state.num_layers == self.num_layers

        new_states = list()
        for i in range(self.num_layers):
            with NoName(input_):
                new_state = self.cells[i](input_, state.get_layer(i, state_direction))
            new_states.append(new_state)
            input_ = new_state[0].refine_names('batch', ...)
            # Note that the last layer doesn't use dropout, following nn.LSTM.
            if i < self.num_layers - 1:
                input_ = self.drop(input_)
        return input_, LstmStatesByLayers(new_states)

    def extra_repr(self):
        return '%d, %d, num_layers=%d' % (self.input_size, self.hidden_size, self.num_layers)


@dataclass
class EmbParams:
    num_embeddings: int
    embedding_dim: int
    dropout: float
    phono_feat_mat: Optional[LT] = None
    special_ids: Optional[Sequence[int]] = None


class SharedEmbedding(nn.Embedding):
    """Shared input and output embedding with dropout."""

    def __init__(self, *args, dropout: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop = nn.Dropout(dropout)

    def project(self, h: FT) -> FT:
        w = self.drop(self.weight)
        return h @ w.t()

    def forward(self, *args, **kwargs) -> FT:
        emb = super().forward(*args, **kwargs)
        return self.drop(emb)

    @classmethod
    def from_params(cls, emb_params: EmbParams) -> SharedEmbedding:
        return cls(emb_params.num_embeddings,
                   emb_params.embedding_dim,
                   dropout=emb_params.dropout)


class PhonoEmbedding(SharedEmbedding):

    def __init__(self,
                 phono_feat_mat: LT,
                 special_ids: Sequence[int],
                 num_embeddings: int,
                 embedding_dim: int, *args, **kwargs):
        num_phones, num_features = phono_feat_mat.shape
        if embedding_dim % num_features > 0:
            raise ValueError(
                f'Embedding size {embedding_dim} cannot be divided by number of phonological features {num_features}.')
        super().__init__(num_embeddings, embedding_dim // num_features, *args, **kwargs)

        self.register_buffer('pfm', phono_feat_mat)
        self.special_weight = nn.Parameter(torch.randn(num_phones, embedding_dim))  # NOTE(j_luo) Use the undivided dim.
        special_mask = torch.zeros(num_phones).bool()
        special_mask[special_ids] = True
        self.register_buffer('special_mask', special_mask)
        self.embedding_dim = embedding_dim

    @classmethod
    def from_params(cls, emb_params: EmbParams) -> PhonoEmbedding:
        return cls(emb_params.phono_feat_mat,
                   emb_params.special_ids,
                   emb_params.num_embeddings,
                   emb_params.embedding_dim,
                   dropout=emb_params.dropout)

    @property
    def char_embedding(self) -> FT:
        """Character embeddings are computed by concatenating all of their relevant phonological feature embeddings."""
        emb = super().forward(self.pfm)
        emb = emb.refine_names(..., 'phono_emb')
        nh = NameHelper()
        emb = nh.flatten(emb, ['phono_feat', 'phono_emb'], 'emb')
        return torch.where(self.special_mask.view(-1, 1), self.special_weight, emb)

    def forward(self, input_: LT) -> FT:
        with NoName(self.char_embedding, input_):
            return self.char_embedding[input_]

    def project(self, h: FT) -> FT:
        return h @ self.char_embedding.t()


def get_embedding(emb_params: EmbParams) -> Union[PhonoEmbedding, SharedEmbedding]:
    emb_cls = SharedEmbedding if emb_params.phono_feat_mat is None else PhonoEmbedding
    embedding = emb_cls.from_params(emb_params)
    return embedding


class GlobalAttention(nn.Module):

    def __init__(self,
                 input_src_size: int,
                 input_tgt_size: int,
                 dropout: float = 0.0):
        super(GlobalAttention, self).__init__()

        self.input_src_size = input_src_size
        self.input_tgt_size = input_tgt_size

        self.Wa = nn.Parameter(torch.Tensor(input_src_size, input_tgt_size))
        torch.nn.init.xavier_normal_(self.Wa)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                h_t: FT,
                h_s: FT,
                mask_src: BT) -> Tuple[FT, FT]:
        sl, bs, ds = h_s.size()
        dt = h_t.shape[-1]
        # FIXME(j_luo) Cache this
        with NoName(h_s):
            Wh_s = self.drop(h_s).reshape(sl * bs, -1).mm(self.Wa).view(sl, bs, -1)

        with NoName(h_t):
            scores = (Wh_s * h_t).sum(dim=-1)

        scores = torch.where(mask_src, scores, torch.full_like(scores, -9999.9))
        almt_distr = nn.functional.log_softmax(scores, dim=0).exp()  # sl x bs
        with NoName(almt_distr):
            ctx = (almt_distr.unsqueeze(dim=-1) * h_s).sum(dim=0)  # bs x d
        almt_distr = almt_distr.t()
        return almt_distr, ctx

    def extra_repr(self):
        return 'src=%d, tgt=%d' % (self.input_src_size, self.input_tgt_size)


class NormControlledResidual(nn.Module):

    def __init__(self, norms_or_ratios=None, multiplier=1.0, control_mode=None):
        super().__init__()

        assert control_mode in ['none', 'relative', 'absolute']

        self.control_mode = control_mode
        self.norms_or_ratios = None
        if self.control_mode in ['relative', 'absolute']:
            self.norms_or_ratios = norms_or_ratios
            if self.control_mode == 'relative':
                assert self.norms_or_ratios[0] == 1.0

        self.multiplier = multiplier

    def anneal_ratio(self):
        if self.control_mode == 'relative':
            new_ratios = [self.norms_or_ratios[0]]
            for r in self.norms_or_ratios[1:]:
                r = min(r * self.multiplier, 1.0)
                new_ratios.append(r)
            self.norms_or_ratios = new_ratios
            logging.debug('Ratios are now [%s]' % (', '.join(map(lambda f: '%.2f' % f, self.norms_or_ratios))))

    def forward(self, *inputs):
        if self.control_mode == 'none':
            output = sum(inputs)
        else:
            assert len(inputs) == len(self.norms_or_ratios)
            outs = list()
            if self.control_mode == 'absolute':
                for inp, norm in zip(inputs, self.norms_or_ratios):
                    if norm >= 0.0:  # NOTE(j_luo) a negative value means no control applied
                        outs.append(normalize(inp, dim=-1) * norm)
                    else:
                        outs.append(inp)
            else:
                outs.append(inputs[0])
                norm_base = inputs[0].norm(dim=-1, keepdim=True)
                for inp, ratio in zip(inputs[1:], self.norms_or_ratios[1:]):
                    if ratio >= 0.0:  # NOTE(j_luo) same here
                        norm_actual = inp.norm(dim=-1, keepdim=True)
                        max_norm = norm_base * ratio
                        too_big = norm_actual > max_norm
                        adjusted_norm = torch.where(too_big, max_norm, norm_actual)
                        outs.append(normalize(inp, dim=-1) * adjusted_norm)
                    else:
                        outs.append(inp)
            output = sum(outs)
        return output


class LanguageEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 unseen_idx: Optional[int] = None,
                 mode: str = 'random',
                 dropout: float = 0.0, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.unseen_idx = unseen_idx
        assert mode in ['random', 'mean']
        self.mode = mode
        self.drop = nn.Dropout(dropout)

    def forward(self, index: int) -> FT:
        if index == self.unseen_idx:
            if self.mode == 'random':
                ret = self.weight[index]
            else:
                ret = (self.weight.sum(dim=0) - self.weight[index]) / (self.num_embeddings - 1)
        else:
            ret = self.weight[index]
        return self.drop(ret)

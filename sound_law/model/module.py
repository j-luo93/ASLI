from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import normalize

from dev_misc import BT, FT, LT, get_zeros
from dev_misc.devlib.named_tensor import NoName
from sound_law.model.lstm_state import LstmStatesByLayers, LstmStateTuple

LstmOutputsByLayers = Tuple[FT, LstmStatesByLayers]
LstmOutputTuple = Tuple[FT, LstmStateTuple]


class SharedEmbedding(nn.Embedding):
    """Shared input and output embedding."""

    def project(self, h: FT) -> FT:
        return h @ self.weight.t()


class MultiLayerLSTMCell(nn.Module):
    """An LSTM cell with multiple layers."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)

        cells = [nn.LSTMCell(input_size, hidden_size)]
        for _ in range(self.num_layers - 1):
            cells.append(nn.LSTMCell(hidden_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, input_: FT, state: LstmStatesByLayers) -> LstmOutputsByLayers:
        assert state.num_layers == self.num_layers

        new_states = list()
        for i in range(self.num_layers):
            # Note that the last layer doesn't use dropout.
            input_ = self.drop(input_)
            with NoName(input_):
                new_state = self.cells[i](input_, state.get_layer(i))
            new_states.append(new_state)
            input_ = new_state[0].refine_names('batch', ...)
        return input_, LstmStatesByLayers(new_states)

    def extra_repr(self):
        return '%d, %d, num_layers=%d' % (self.input_size, self.hidden_size, self.num_layers)


class LstmCellWithEmbedding(nn.Module):
    """An LSTM cell on top of an embedding layer."""

    def __init__(self,
                 num_embeddings: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 embedding: Optional[nn.Module] = None):
        super().__init__()

        self.embedding = embedding or SharedEmbedding(num_embeddings, input_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = MultiLayerLSTMCell(input_size, hidden_size, num_layers, dropout=dropout)

    def embed(self, input_: LT) -> FT:
        return self.embedding(input_)

    def forward(self, input_: LT, init_state: Optional[LstmStatesByLayers] = None) -> LstmOutputsByLayers:
        emb = self.embed(input_)
        emb = self.dropout(emb)

        batch_size = input_.size('batch')
        init_state = init_state or LstmStateTuple.zero_state(self.lstm.num_layers,
                                                             batch_size,
                                                             self.lstm.hidden_size)
        output, state = self.lstm(emb, init_state)
        return output, state


class LstmDecoder(LstmCellWithEmbedding):
    """A decoder that unrolls the LSTM decoding procedure into steps."""

    def forward(self,
                sot_id: int,  # "start-of-token"
                init_state: LstmStatesByLayers,
                max_length: Optional[int] = None,
                target: Optional[LT] = None) -> FT:
        if self.training:
            assert target is not None
            assert target.names[1] == 'batch'
            assert len(target.shape) == 2
        if max_length is None:
            max_length = target.size("pos")

        state = init_state
        batch_size = init_state.batch_size
        log_probs = list()
        input_ = torch.full([batch_size], sot_id, dtype=torch.long).rename('batch').to(init_state.device)
        for l in range(max_length):
            output, state = super().forward(input_, state)
            logit = self.embedding.project(output)
            log_prob = logit.log_softmax(dim=-1)
            log_probs.append(log_prob)

            if self.training:
                input_ = target[l]
            else:
                input_ = log_prob.max(dim=-1)[1]

        with NoName(*log_probs):
            log_probs = torch.stack(log_probs, dim=0).refine_names('pos', 'batch', 'unit')
        return log_probs


class LstmEncoder(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 embedding: Optional[nn.Module] = None):
        super().__init__()
        self.embedding = embedding or SharedEmbedding(num_embeddings, input_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)

    def forward(self, input_: LT) -> LstmOutputTuple:
        # FIXME(j_luo) what happened to paddings?
        batch_size = input_.size('batch')
        emb = self.embedding(input_)
        with NoName(emb):
            output, state = self.lstm(emb)
        return output, LstmStateTuple(state, bidirectional=self.lstm.bidirectional)

# class GlobalAttention(nn.Module):

#     def __init__(self,
#                  input_src_size: int,
#                  input_tgt_size: int,
#                  dropout: float = 0.0):
#         super(GlobalAttention, self).__init__()

#         self.input_src_size = input_src_size
#         self.input_tgt_size = input_tgt_size
#         self.dropout = dropout

#         self.Wa = nn.Parameter(torch.Tensor(input_src_size, input_tgt_size))
#         self.drop = nn.Dropout(self.dropout)

#     @cache(full=False)
#     def _get_Wh_s(self, h_s):
#         bs, l, _ = h_s.shape
#         Wh_s = self.drop(h_s).reshape(bs * l, -1).mm(self.Wa).view(bs, l, -1)
#         return Wh_s

#     def forward(self,
#                 h_t: FT,
#                 h_s: FT,
#                 mask_src: BT):
#         bs, sl, ds = h_s.size()
#         dt = h_t.shape[-1]
#         ...  # FIXME(j_luo) fill in this
#         Wh_s = self._get_Wh_s(h_s)  # bs x sl x dt

#         scores = Wh_s.matmul(self.drop(h_t).unsqueeze(dim=-1)).squeeze(dim=-1)  # bs x sl

#         scores = torch.where(mask_src, scores, torch.full_like(scores, -9999.9))
#         almt_distr = nn.functional.log_softmax(scores, dim=-1).exp()  # bs x sl
#         return almt_distr

#     def extra_repr(self):
#         return 'src=%d, tgt=%d' % (self.input_src_size, self.input_tgt_size)

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dev_misc import FT, LT
from dev_misc.devlib.named_tensor import NoName

from .lstm_state import LstmStateTuple
from .module import EmbParams, LstmParams, get_embedding

LstmOutputTuple = Tuple[FT, LstmStateTuple]


class LstmEncoder(nn.Module):

    def __init__(self, embedding: nn.Module, lstm: nn.LSTM):
        super().__init__()
        self.embedding = embedding
        self.lstm = lstm

    @classmethod
    def from_params(cls, emb_params: EmbParams, lstm_params: LstmParams) -> LstmEncoder:
        embedding = get_embedding(emb_params)
        lstm = nn.LSTM(
            lstm_params.input_size,
            lstm_params.hidden_size,
            lstm_params.num_layers,
            bidirectional=lstm_params.bidirectional,
            dropout=lstm_params.dropout)
        return cls(embedding, lstm)

    def forward(self, input_: LT, lengths: LT) -> Tuple[FT, LstmOutputTuple]:
        emb = self.embedding(input_)
        with NoName(emb, lengths):
            packed_emb = pack_padded_sequence(emb, lengths, enforce_sorted=False)
            output, state = self.lstm(packed_emb)
            output = pad_packed_sequence(output)[0]
        return emb, (output, LstmStateTuple(state, bidirectional=self.lstm.bidirectional))

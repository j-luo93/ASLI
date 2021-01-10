from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dev_misc import FT, LT
from dev_misc.devlib.named_tensor import NoName

from .lstm_state import LstmStateTuple
from .module import CharEmbedding, EmbParams, LstmParams, get_embedding

LstmOutputTuple = Tuple[FT, LstmStateTuple]


class LstmEncoder(nn.Module):

    def __init__(self, embedding: nn.Module, lstm: nn.LSTM, dropout: float = 0.0):
        super().__init__()
        self.embedding = embedding
        self.lstm = lstm
        self.drop = nn.Dropout(dropout)

    @classmethod
    def from_params(cls, emb_params: EmbParams, lstm_params: LstmParams) -> LstmEncoder:
        embedding = get_embedding(emb_params)
        lstm = nn.LSTM(
            lstm_params.input_size,
            lstm_params.hidden_size,
            lstm_params.num_layers,
            bidirectional=lstm_params.bidirectional,
            dropout=lstm_params.dropout)
        return cls(embedding, lstm, lstm_params.dropout)

    def forward(self, input_: LT, lengths: LT) -> Tuple[FT, LstmOutputTuple]:
        emb = self.embedding(input_)
        with NoName(emb, lengths):
            packed_emb = pack_padded_sequence(emb, lengths, enforce_sorted=False)
            output, state = self.lstm(packed_emb)
            output = pad_packed_sequence(output)[0]
            output = self.drop(output)  # Dropout after last output, different from the behavior for nn.LSTM.
        return emb, (output, LstmStateTuple(state, bidirectional=self.lstm.bidirectional))


@dataclass
class CnnParams:
    hidden_size: int
    kernel_sizes: Tuple[int, ...]
    dropout: float
    stride: int = 1


class CnnEncoder(nn.Module):

    def __init__(self,
                 embedding: CharEmbedding,
                 conv_layers: List[nn.Conv1d],
                 W_output: nn.Linear,
                 dropout: float):
        super().__init__()
        self.embedding = embedding
        self.conv_layers = nn.ModuleList(conv_layers)
        self.W_output = W_output
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_params(cls, emb_params: EmbParams, cnn_params: CnnParams) -> CnnEncoder:
        input_size = emb_params.embedding_dim
        hidden_size = cnn_params.hidden_size
        kernel_sizes = cnn_params.kernel_sizes
        stride = cnn_params.stride

        assert input_size == hidden_size  # the layers' dimensionalities rely on this assumption
        # we want all the convolutional layers to create outputs with the same length. It's easiest to calculate the padding to do so when the kernel sizes are odd, so for now we only support odd kernel sizes.
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        embedding = get_embedding(emb_params)

        conv_layers = list()
        for kernel_size in kernel_sizes:
            # we pad the layers so that the output for each layer is of length seq_length
            padding = int((kernel_size - 1) / 2)
            layer = nn.Conv1d(input_size, hidden_size,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
            conv_layers.append(layer)

        # the input size of this output projection layer depends on the number of convolutional modules
        n_conv = len(kernel_sizes)
        W_output = nn.Linear(n_conv * hidden_size, 2 * hidden_size)
        return cls(embedding, conv_layers, W_output, cnn_params.dropout)

    def forward(self, input_: LT, lengths: LT) -> Tuple[FT, LstmOutputTuple]:
        # input_: seq_length x batch_size
        # note that input_size == hidden_size
        # define n_conv as the number of parallel convolutional layers
        emb = self.embedding(input_)  # seq_length x batch_size x input_size

        with NoName(emb, lengths):
            reshaped_emb = emb.permute(1, 2, 0)  # reshape to batch_size x input_size x seq_length for CNN input
            conv_outputs = [self.dropout(F.relu(conv(reshaped_emb))) for conv in self.conv_layers]
            # each conv layer's output is batch_size x hidden_size x seq_length

            # stack the CNN outputs on the hidden_size dimension
            x = torch.cat(conv_outputs, dim=1)  # batch_size x n_conv*hidden_size x seq_length
            x = x.permute(2, 0, 1)  # seq_length x batch_size x n_conv*hidden_size

            # project the concatenated convolutional layer outputs into 2*hidden_size dimensions so that `output` looks as though it were the states of a bidirectional lstm
            output = self.W_output(x)  # seq_length x batch_size x 2*hidden_size
            # we don't try to reconstruct the state, so we just pass (None, None)
        return emb, (output, (None, None))

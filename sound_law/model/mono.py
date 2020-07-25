"""
This file contains monolingual models.
"""
import torch
import torch.nn as nn

from dev_misc import add_argument, g, get_zeros
from sound_law.data.data_loader import MonoBatch

from .module import LstmDecoder, LstmEncoder


class MonoModel(nn.Module):

    add_argument('char_emb_size', default=256, dtype=int, msg='Embedding size for characters (as input).')
    add_argument('hidden_size', default=256, dtype=int, msg='Hidden size for LSTM states.')
    add_argument('num_layers', default=1, dtype=int, msg='Number of LSTM layers.')
    add_argument('dropout', default=0.2, dtype=int, msg='Dropout rate.')

    def __init__(self, num_src_chars: int, num_tgt_chars: int):
        super().__init__()
        self.encoder = LstmEncoder(num_src_chars,
                                   g.char_emb_size,
                                   g.hidden_size,
                                   g.num_layers,
                                   dropout=g.dropout,
                                   bidirectional=True)
        self.decoder = LstmDecoder(num_tgt_chars,
                                   g.char_emb_size,
                                   g.hidden_size,
                                   g.num_layers,
                                   dropout=g.dropout)

    def forward(self, batch: MonoBatch) -> FT:
        output, state = self.encoder(batch.src_seq)
        log_probs = self.decoder(batch.tgt_seq, state)
        loss = log_probs.sum()  # FIXME(j_luo) fill in this: gather
        return loss

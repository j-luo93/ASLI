"""
This file contains models for one pair of src-tgt languags.
"""

import torch.nn as nn

from dev_misc import FT, LT, add_argument, g, get_zeros
from dev_misc.devlib.named_tensor import Rename
from sound_law.data.data_loader import OnePairBatch, PaddedUnitSeqs
from sound_law.data.dataset import SOT_ID

from .module import LstmDecoderWithAttention, LstmEncoder


class OnePairModel(nn.Module):

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
        self.decoder = LstmDecoderWithAttention(num_tgt_chars,
                                                g.char_emb_size,
                                                g.hidden_size * 2,
                                                g.hidden_size,
                                                g.num_layers,
                                                dropout=g.dropout)

    def _get_log_probs(self, batch: OnePairBatch, use_target: bool = True, max_length: int = None) -> FT:
        output, state = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        target = batch.tgt_seqs.ids if use_target else None
        log_probs = self.decoder(SOT_ID, output, batch.src_seqs.paddings,
                                 target=target,
                                 max_length=max_length)
        return log_probs

    def forward(self, batch: OnePairBatch) -> FT:
        log_probs = self._get_log_probs(batch)
        loss = -log_probs.gather('unit', batch.tgt_seqs.ids)
        loss = loss * batch.tgt_seqs.paddings.float()
        return loss

    def get_scores(self, batch: OnePairBatch, tgt_vocab_seqs: PaddedUnitSeqs) -> FT:
        """Given a batch and a list of target tokens (provided as id sequences), return scores produced by the model."""
        assert not self.training
        max_length = tgt_vocab_seqs.ids.size('pos')
        log_probs = self._get_log_probs(batch, use_target=False, max_length=max_length)
        with Rename(tgt_vocab_seqs.ids, batch='tgt_vocab'), Rename(tgt_vocab_seqs.paddings, batch='tgt_vocab'):
            unit_scores = log_probs.gather('unit', tgt_vocab_seqs.ids)
            unit_scores = unit_scores * tgt_vocab_seqs.paddings.float().align_as(unit_scores)
        scores = unit_scores.sum('pos')
        return scores

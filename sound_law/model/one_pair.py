"""
This file contains models for one pair of src-tgt languags.
"""

from typing import Optional, Sequence, Tuple

import torch.nn as nn

from dev_misc import FT, LT, add_argument, g, get_zeros
from dev_misc.devlib.named_tensor import Rename
from sound_law.data.data_loader import OnePairBatch, PaddedUnitSeqs
from sound_law.data.dataset import SOT_ID

from .decoder import DecParams, LstmDecoder
from .encoder import LstmEncoder
from .module import EmbParams, LstmParams


class OnePairModel(nn.Module):

    add_argument('char_emb_size', default=256, dtype=int, msg='Embedding size for characters (as input).')
    add_argument('hidden_size', default=256, dtype=int, msg='Hidden size for LSTM states.')
    add_argument('num_layers', default=1, dtype=int, msg='Number of LSTM layers.')
    add_argument('dropout', default=0.2, dtype=int, msg='Dropout rate.')
    add_argument('norms_or_ratios', default=(1.0, 0.2), nargs=2, dtype=float,
                 msg='Norms or ratios of norms for the norm-controlled residual module.')
    add_argument('control_mode', default='relative', dtype=str, choices=['relative', 'absolute', 'none'],
                 msg='Control mode for the norm-controlled residual module.')

    def __init__(self, num_src_chars: int, num_tgt_chars: int,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):

        super().__init__()

        def get_emb_params(num_chars: int) -> EmbParams:
            return EmbParams(num_chars, g.char_emb_size, g.dropout,
                             phono_feat_mat=phono_feat_mat,
                             special_ids=special_ids)

        def get_lstm_params(bidirectional: bool) -> LstmParams:
            return LstmParams(g.char_emb_size, g.hidden_size,
                              g.num_layers, g.dropout,
                              bidirectional=bidirectional)

        enc_emb_params = get_emb_params(num_src_chars)
        enc_lstm_params = get_lstm_params(True)
        self.encoder = LstmEncoder.from_params(enc_emb_params, enc_lstm_params)

        if g.share_src_tgt_abc:
            dec_emb_params = None
            dec_embedding = self.encoder.embedding
        else:
            dec_emb_params = get_emb_params(num_tgt_chars)
            dec_embedding = None
        dec_lstm_params = get_lstm_params(False)
        dec_params = DecParams(dec_lstm_params,
                               g.hidden_size * 2,  # Bidirectional outputs.
                               g.hidden_size,
                               g.norms_or_ratios,
                               g.control_mode,
                               emb_params=dec_emb_params)
        self.decoder = LstmDecoder.from_params(dec_params,
                                               embedding=dec_embedding)

    def forward(self, batch: OnePairBatch, use_target: bool = True, max_length: int = None) -> Tuple[FT, FT]:
        src_emb, (output, state) = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        target = batch.tgt_seqs.ids if use_target else None
        log_probs, almt_distrs = self.decoder(SOT_ID, src_emb,
                                              output, batch.src_seqs.paddings,
                                              max_length=max_length,
                                              target=target)
        return log_probs, almt_distrs

    def get_scores(self, batch: OnePairBatch, tgt_vocab_seqs: PaddedUnitSeqs) -> FT:
        """Given a batch and a list of target tokens (provided as id sequences), return scores produced by the model."""
        assert not self.training
        max_length = tgt_vocab_seqs.ids.size('pos')
        log_probs, _ = self.forward(batch, use_target=False, max_length=max_length)
        with Rename(tgt_vocab_seqs.ids, batch='tgt_vocab'), Rename(tgt_vocab_seqs.paddings, batch='tgt_vocab'):
            unit_scores = log_probs.gather('unit', tgt_vocab_seqs.ids)
            unit_scores = unit_scores * tgt_vocab_seqs.paddings.float().align_as(unit_scores)
        scores = unit_scores.sum('pos')
        return scores

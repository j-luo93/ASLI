"""
This file contains models for one pair of src-tgt languags.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dev_misc import FT, LT, add_argument, add_condition, g, get_zeros
from dev_misc.devlib.named_tensor import NoName, Rename
from dev_misc.utils import pbar
from sound_law.data.data_loader import OnePairBatch, PaddedUnitSeqs
from sound_law.data.dataset import SOT_ID

from .decoder import DecParams, LstmDecoder
from .encoder import CnnEncoder, CnnParams, LstmEncoder
from .module import EmbParams, LstmParams


class OnePairModel(nn.Module):

    add_argument('char_emb_size', default=256, dtype=int, msg='Embedding size for characters (as input).')
    add_argument('hidden_size', default=256, dtype=int, msg='Hidden size for LSTM states.')
    add_argument('num_layers', default=1, dtype=int, msg='Number of LSTM layers.')
    add_argument('dropout', default=0.2, dtype=float, msg='Dropout rate.')
    add_argument('norms_or_ratios', default=(1.0, 0.2), nargs=2, dtype=float,
                 msg='Norms or ratios of norms for the norm-controlled residual module.')
    add_argument('control_mode', default='relative', dtype=str, choices=['relative', 'absolute', 'none'],
                 msg='Control mode for the norm-controlled residual module.')
    add_argument('model_encoder_type', dtype=str, default='lstm', choices=['lstm', 'cnn'], msg='Which encoder to use.')
    add_argument('kernel_sizes', dtype=int, nargs='+', default=(3, 5, 7),
                 msg='What kernel sizes to use for the CNN Encoder (can include repeats).')

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
        if g.model_encoder_type == 'lstm':
            enc_lstm_params = get_lstm_params(True)
            self.encoder = LstmEncoder.from_params(enc_emb_params, enc_lstm_params)
        else:
            cnn_params = CnnParams(g.hidden_size, g.kernel_sizes, g.dropout)
            self.encoder = CnnEncoder.from_params(enc_emb_params, cnn_params)

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

    # def old_get_scores(self, batch: OnePairBatch, tgt_vocab_seqs: PaddedUnitSeqs) -> FT:
    #     """Given a batch and a list of target tokens (provided as id sequences), return scores produced by the model."""
    #     assert not self.training
    #     max_length = tgt_vocab_seqs.ids.size('pos')
    #     log_probs, _ = self.forward(batch, use_target=False, max_length=max_length)
    #     with Rename(tgt_vocab_seqs.ids, batch='tgt_vocab'), Rename(tgt_vocab_seqs.paddings, batch='tgt_vocab'):
    #         unit_scores = log_probs.gather('unit', tgt_vocab_seqs.ids)
    #         unit_scores = unit_scores * tgt_vocab_seqs.paddings.float().align_as(unit_scores)
    #     scores = unit_scores.sum('pos')
    #     return scores

    def get_scores(self, batch: OnePairBatch, tgt_vocab_seqs: PaddedUnitSeqs, chunk_size: int = 100) -> FT:
        """Given a batch and a list of target tokens (provided as id sequences), return scores produced by the model."""
        src_emb, (output, state) = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        src_emb = src_emb.refine_names('pos', 'batch', 'src_emb')
        output = output.refine_names('pos', 'batch', 'output')
        batch_size = src_emb.size('batch')

        def create_chunk(size, base, old_chunk, interleave: bool = True):
            if not interleave:
                return base.repeat(1, batch_size)

            if old_chunk is not None and old_chunk.size('batch') == batch_size * size:
                return old_chunk

            new_chunk = torch.repeat_interleave(base, size, dim='batch')
            return new_chunk

        chunk_src_emb = None
        chunk_output = None
        chunk_src_paddings = None
        scores = list()
        for split in pbar(tgt_vocab_seqs.split(chunk_size), desc='Get scores: chunk'):
            split: PaddedUnitSeqs
            bs_split = len(split)
            chunk_src_emb = create_chunk(bs_split, src_emb, chunk_src_emb)
            chunk_output = create_chunk(bs_split, output, chunk_output)
            chunk_src_paddings = create_chunk(bs_split, batch.src_seqs.paddings, chunk_src_paddings)
            chunk_target = create_chunk(None, split.ids, None, interleave=False)
            chunk_tgt_paddings = create_chunk(None, split.paddings, None, interleave=False)
            chunk_log_probs, _ = self.decoder(SOT_ID, chunk_src_emb,
                                              chunk_output, chunk_src_paddings,
                                              target=chunk_target)
            chunk_scores = chunk_log_probs.gather('unit', chunk_target)
            chunk_scores = (chunk_scores * chunk_tgt_paddings).sum('pos')
            with NoName(chunk_scores):
                scores.append(chunk_scores.view(batch_size, bs_split).refine_names('batch', 'tgt_vocab'))
        scores = torch.cat(scores, dim='tgt_vocab')
        return scores

"""
This file contains models for one pair of src-tgt languags.
"""

from abc import abstractmethod
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from dev_misc import FT, LT, add_argument, add_condition, g, get_zeros
from dev_misc.devlib.named_tensor import NoName, Rename
from dev_misc.utils import pbar
from sound_law.data.data_loader import (OnePairBatch, PaddedUnitSeqs,
                                        SourceOnlyBatch)
from sound_law.data.dataset import SOT_ID

from .decoder import DecParams, Hypotheses, LstmDecoder
from .encoder import CnnEncoder, CnnParams, LstmEncoder
from .module import EmbParams, LstmParams


class BaseModel(nn.Module):

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
    add_argument('beam_size', dtype=int, default=1, msg='Beam size.')
    add_argument('separate_output', dtype=bool, default=False,
                 msg='Flag to use a separate set of params for output embeddings.')

    def __init__(self, num_src_chars: int, num_tgt_chars: int,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):

        super().__init__()

        def get_emb_params(num_chars: int) -> EmbParams:
            return EmbParams(num_chars, g.char_emb_size, g.dropout,
                             phono_feat_mat=phono_feat_mat,
                             special_ids=special_ids,
                             separate_output=g.separate_output)

        def get_lstm_params(input_size: int, bidirectional: bool) -> LstmParams:
            return LstmParams(input_size, g.hidden_size,
                              g.num_layers, g.dropout,
                              bidirectional=bidirectional)

        enc_emb_params = get_emb_params(num_src_chars)
        if g.model_encoder_type == 'lstm':
            enc_lstm_params = get_lstm_params(g.char_emb_size, True)
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
        # NOTE(j_luo) Input size is the sum of `g.char_emb_size` and `g.hidden_size` if input feeding is used.
        dec_input_size = g.char_emb_size + (g.hidden_size if g.input_feeding else 0)
        dec_lstm_params = get_lstm_params(dec_input_size, False)
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
        lang_emb = self._prepare_lang_emb(batch)
        log_probs, almt_distrs = self.decoder(SOT_ID, src_emb,
                                              output, batch.src_seqs.paddings,
                                              max_length=max_length,
                                              target=target,
                                              lang_emb=lang_emb)
        return log_probs, almt_distrs

    def get_scores(self, batch: OnePairBatch, tgt_vocab_seqs: PaddedUnitSeqs, chunk_size: int = 100) -> FT:
        """Given a batch and a list of target tokens (provided as id sequences), return scores produced by the model."""
        src_emb, (output, state) = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        src_emb = src_emb.refine_names('pos', 'batch', 'src_emb')
        output = output.refine_names('pos', 'batch', 'output')
        batch_size = src_emb.size('batch')
        lang_emb = self._prepare_lang_emb(batch)

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
                                              target=chunk_target,
                                              lang_emb=lang_emb)
            chunk_scores = chunk_log_probs.gather('unit', chunk_target)
            chunk_scores = (chunk_scores * chunk_tgt_paddings).sum('pos')
            with NoName(chunk_scores):
                scores.append(chunk_scores.view(batch_size, bs_split).refine_names('batch', 'tgt_vocab'))
        scores = torch.cat(scores, dim='tgt_vocab')
        return scores

    def predict(self, batch: Union[SourceOnlyBatch, OnePairBatch]) -> Hypotheses:
        src_emb, (output, state) = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        src_emb = src_emb.refine_names('pos', 'batch', 'src_emb')
        output = output.refine_names('pos', 'batch', 'output')

        lang_emb = self._prepare_lang_emb(batch)
        hyps = self.decoder.search(SOT_ID, src_emb, output,
                                   batch.src_seqs.paddings,
                                   batch.src_seqs.lengths,
                                   g.beam_size,
                                   lang_emb=lang_emb)
        return hyps

    @abstractmethod
    def _prepare_lang_emb(self, batch: OnePairBatch) -> FT: ...

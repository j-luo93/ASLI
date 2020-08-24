from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dev_misc import BT, FT, LT
from dev_misc.devlib.named_tensor import (NameHelper, NoName, duplicate,
                                          get_named_range)
from dev_misc.utils import ScopedCache

from .beam_searcher import BaseBeamSearcher
from .lstm_state import LstmStatesByLayers
from .module import (EmbParams, GlobalAttention, LanguageEmbedding, LstmParams,
                     MultiLayerLSTMCell, NormControlledResidual,
                     SharedEmbedding, get_embedding)


@dataclass
class DecParams:
    lstm_params: LstmParams
    # Sizes for attention.
    src_hidden_size: int
    tgt_hidden_size: int
    # Params for residual.
    norms_or_ratios: Tuple[float]
    control_mode: str
    # This is optional due to potential module sharing with encoder.
    emb_params: Optional[EmbParams] = None


@dataclass
class Beam:
    step: int
    batch_size: int
    beam_size: int
    accum_scores: FT
    tokens: LT  # Predicted tokens from last step.
    lstm_state: LstmStatesByLayers  # Last LSTM state.
    # Constant inputs.
    src_emb: FT
    src_outputs: FT
    src_paddings: BT
    lang_emb: Optional[FT] = None


@dataclass
class Candidates:
    log_probs: FT
    state: LstmStatesByLayers  # The LSTM state that is mapped to `log_probs`.


class LstmDecoder(nn.Module, BaseBeamSearcher):
    """A decoder that unrolls the LSTM decoding procedure by steps."""

    def __init__(self,
                 char_emb: SharedEmbedding,
                 cell: MultiLayerLSTMCell,
                 attn: GlobalAttention,
                 hidden: nn.Linear,
                 nc_residual: NormControlledResidual):
        super().__init__()
        self.char_emb = char_emb
        self.cell = cell
        self.attn = attn
        self.hidden = hidden
        self.nc_residual = nc_residual

    @classmethod
    def from_params(cls,
                    dec_params: DecParams,
                    embedding: Optional[SharedEmbedding] = None) -> LstmDecoder:
        emb_params = dec_params.emb_params
        lstm_params = dec_params.lstm_params
        if emb_params is None and embedding is None:
            raise ValueError('Must specify either `emb_params` or `embedding`.')
        embedding_dim = emb_params.embedding_dim if emb_params is not None else embedding.embedding_dim
        if embedding_dim != lstm_params.input_size:
            raise ValueError(
                f'Expect equal values, but got {emb_params.embedding_dim} and {lstm_params.input_size}.')

        char_emb = get_embedding(emb_params) if embedding is None else embedding
        cell = MultiLayerLSTMCell.from_params(lstm_params)
        attn = GlobalAttention(dec_params.src_hidden_size,
                               dec_params.tgt_hidden_size)
        hidden = nn.Linear(
            dec_params.src_hidden_size + dec_params.tgt_hidden_size,
            dec_params.tgt_hidden_size)
        nc_residual = NormControlledResidual(
            norms_or_ratios=dec_params.norms_or_ratios,
            control_mode=dec_params.control_mode)

        return LstmDecoder(char_emb, cell, attn, hidden, nc_residual)

    def forward(self,
                sot_id: int,
                src_emb: FT,
                src_outputs: FT,
                mask_src: BT,
                max_length: Optional[int] = None,
                target: Optional[LT] = None,
                lang_emb: Optional[FT] = None) -> Tuple[FT, FT]:
        # Prepare inputs.
        max_length = self._get_max_length(max_length, target)
        batch_size = mask_src.size('batch')
        input_ = self._prepare_first_input(sot_id, batch_size, mask_src.device)
        state = LstmStatesByLayers.zero_state(
            self.cell.num_layers,
            batch_size,
            self.attn.input_tgt_size,
            bidirectional=False)

        # Main loop.
        log_probs = list()
        almt_distrs = list()
        with ScopedCache('Wh_s'):
            for l in range(max_length):
                state, log_prob, almt_distr = self._forward_step(
                    input_, src_emb, state, src_outputs, mask_src,
                    lang_emb=lang_emb)
                if target is None:
                    input_ = log_prob.max(dim=-1)[1].rename('batch')
                else:
                    input_ = target[l]

                log_probs.append(log_prob)
                almt_distrs.append(almt_distr)

        # Prepare outputs.
        with NoName(*log_probs), NoName(*almt_distrs):
            log_probs = torch.stack(log_probs).rename('pos', 'batch', 'unit')
            almt_distrs = torch.stack(almt_distrs).rename('tgt_pos', 'batch', 'src_pos')
        return log_probs, almt_distrs

    def _get_max_length(self, max_length: Optional[int], target: Optional[LT]) -> int:
        if self.training:
            assert target is not None
            assert target.names[1] == 'batch'
            assert len(target.shape) == 2
        if max_length is None:
            max_length = target.size("pos")
        return max_length

    def _prepare_first_input(self, sot_id: int, batch_size: int, device: torch.device) -> FT:
        input_ = torch.full([batch_size], sot_id, dtype=torch.long).rename('batch').to(device)
        return input_

    def _forward_step(self,
                      input_: LT,
                      src_emb: FT,
                      state: LstmStatesByLayers,
                      src_states: FT,
                      mask_src: BT,
                      lang_emb: Optional[FT] = None) -> Tuple[FT, FT, FT]:
        emb = self.char_emb(input_)
        if lang_emb is not None:
            emb = emb + lang_emb
        # TODO(j_luo) add dropout.
        hid_rnn, next_state = self.cell(emb, state)
        almt, ctx = self.attn.forward(hid_rnn, src_states, mask_src)
        with NoName(hid_rnn, ctx):
            cat = torch.cat([hid_rnn, ctx], dim=-1)
        hid_cat = self.hidden(cat)

        with NoName(src_emb, hid_cat, almt):
            ctx_emb = (src_emb * almt.t().unsqueeze(dim=-1)).sum(dim=0)
            hid_res = self.nc_residual(ctx_emb, hid_cat)

        logit = self.char_emb.project(hid_res)
        log_prob = logit.log_softmax(dim=-1).refine_names('batch', 'unit')

        return next_state, log_prob, almt

    def is_finished(self, beam: Beam, max_lengths: LT) -> BT:
        return beam.step >= max_lengths

    def get_next_candidates(self, beam: Beam) -> Candidates:
        nh = NameHelper()

        def collapse_beam(orig, is_lstm_state: bool = False):

            def wrapped(tensor):
                return nh.flatten(tensor, ['batch', 'beam'], 'BB').rename(BB='batch')

            if is_lstm_state:
                return orig.apply(wrapped)
            return wrapped(orig)

        state, log_probs, almt = self._forward_step(collapse_beam(beam.tokens),
                                                    collapse_beam(beam.src_emb),
                                                    collapse_beam(beam.lstm_state, is_lstm_state=True),
                                                    collapse_beam(beam.src_outputs),
                                                    collapse_beam(beam.src_paddings),
                                                    lang_emb=beam.lang_emb)

        def unflatten(orig, is_lstm_state: bool = False):
            def wrapped(tensor):
                return nh.unflatten(tensor.rename(batch='BB'), 'BB', ['batch', 'beam'])

            if is_lstm_state:
                return orig.apply(wrapped)
            return wrapped(orig)

        log_probs = unflatten(log_probs)
        state = unflatten(state, is_lstm_state=True)
        return Candidates(log_probs, state)

    def get_next_beam(self, beam: Beam, cand: Candidates) -> Beam:
        nh = NameHelper()
        accum = cand.log_probs + beam.accum_scores.align_as(cand.log_probs)
        lp = nh.flatten(accum, ['beam', 'unit'], 'BU')
        top_s, top_i = torch.topk(lp, beam.beam_size, dim='BU')
        num_units = accum.size('unit')
        beam_i = top_i // num_units
        tokens = top_i % num_units

        batch_i = get_named_range(beam.batch_size, 'batch')
        batch_i = batch_i.align_as(top_i)

        def retrieve_state(tensor):
            with NoName(tensor, batch_i, beam_i):
                ret = tensor[batch_i, beam_i]
            return ret.refine_names('batch', 'beam', 'hidden')

        state = cand.state.apply(retrieve_state)

        next_beam = Beam(beam.step + 1,
                         beam.batch_size,
                         beam.beam_size,
                         top_s.rename(BU='beam'),
                         tokens.rename(BU='beam'),
                         state,
                         beam.src_emb,
                         beam.src_outputs,
                         beam.src_paddings,
                         beam.lang_emb
                         )
        return next_beam

    def search(self,
               sot_id: int,
               src_emb: FT,
               src_outputs: FT,
               src_paddings: BT,
               src_lengths: LT,
               beam_size: int,
               lang_emb: Optional[FT] = None) -> Beam:
        if beam_size <= 0:
            raise ValueError(f'`beam_size` must be positive.')

        batch_size = src_emb.size('batch')
        tokens = torch.full([batch_size, beam_size], sot_id, dtype=torch.long).to(
            src_emb.device).rename('batch', 'beam')
        accum_scores = torch.zeros_like(tokens)
        lstm_state = LstmStatesByLayers.zero_state(
            self.cell.num_layers,
            batch_size,
            beam_size,
            self.attn.input_tgt_size,
            bidirectional=False,
            names=['batch', 'beam', 'hidden'])

        def expand_beam(orig):
            return duplicate(orig, 'batch', beam_size, 'beam')

        src_emb = expand_beam(src_emb)
        src_outputs = expand_beam(src_outputs)
        src_paddings = expand_beam(src_paddings)
        max_lengths = (src_lengths.float() * 1.5).long()
        max_lengths = expand_beam(max_lengths)
        init_beam = Beam(0, batch_size, beam_size,
                         accum_scores, tokens, lstm_state,
                         src_emb, src_outputs, src_paddings,
                         lang_emb=lang_emb)
        final_beam = super().search(init_beam, max_lengths)
        return final_beam

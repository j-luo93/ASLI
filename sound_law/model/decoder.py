from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dev_misc import BT, FT, LT
from dev_misc.devlib.named_tensor import NoName

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
    # Shared embedding with encoder if specified.
    emb_params: Optional[EmbParams] = None


class LstmDecoder(nn.Module):
    """A decoder that unrolls the LSTM decoding procedure by steps."""

    def __init__(self,
                 char_emb: SharedEmbedding,
                 cell: MultiLayerLSTMCell,
                 attn: GlobalAttention,
                 hidden: nn.Linear,
                 nc_residual: NormControlledResidual,
                 lang_emb: Optional[LanguageEmbedding] = None):
        super().__init__()
        self.char_emb = char_emb
        self.cell = cell
        self.attn = attn
        self.hidden = hidden
        self.nc_residual = nc_residual
        self.lang_emb = lang_emb

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
                lang_id: Optional[int] = None) -> Tuple[FT, FT]:
        # Prepare inputs.
        max_length = self._get_max_length(max_length, target)
        batch_size = mask_src.size('batch')
        lang_emb = None if lang_id is None else self.lang_emb(lang_id)
        input_ = self._prepare_first_input(sot_id, batch_size, mask_src.device)
        state = LstmStatesByLayers.zero_state(
            self.cell.num_layers,
            batch_size,
            self.attn.input_tgt_size,
            bidirectional=False)

        # Main loop.
        log_probs = list()
        almt_distrs = list()
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
        assert lang_emb is None
        hid_rnn, next_state = self.cell(emb, state)
        almt, ctx = self.attn.forward(hid_rnn, src_states, mask_src)
        cat = torch.cat([hid_rnn, ctx], dim=-1)
        hid_cat = self.hidden(cat)

        with NoName(src_emb, hid_cat, almt):
            ctx_emb = (src_emb * almt.t().unsqueeze(dim=-1)).sum(dim=0)
            hid_res = self.nc_residual(ctx_emb, hid_cat)

        logit = self.char_emb.project(hid_res)
        log_prob = logit.log_softmax(dim=-1)

        return next_state, log_prob, almt

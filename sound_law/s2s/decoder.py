from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, NDA, add_argument, g, get_zeros
from dev_misc.devlib.named_tensor import (NameHelper, NoName, duplicate,
                                          get_named_range)
from dev_misc.utils import ScopedCache, handle_sequence_inputs
from sound_law.data.alphabet import Alphabet
from sound_law.data.dataset import EOT_ID
from sound_law.evaluate.edit_dist import translate

from .beam_searcher import BaseBeamSearcher
from .lstm_state import LstmStatesByLayers
from .module import (CharEmbedding, EmbParams, GlobalAttention,
                     LanguageEmbedding, LstmParams, MultiLayerLSTMCell,
                     NormControlledResidual, get_embedding)

try:
    import graphviz
    from graphviz import Digraph
except ModuleNotFoundError:
    pass


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
class BeamConstant:
    """This stores some common inputs that are constant for all steps."""
    src_emb: FT
    src_outputs: FT
    src_paddings: BT
    max_lengths: LT
    lang_emb: Optional[FT] = None


def _stack_beam(lst: List[torch.Tensor], last_name=None):
    new_names = ('batch', 'beam', 'pos')
    if last_name:
        new_names += (last_name,)
    with NoName(*lst):
        # NOTE(j_luo) Set dim = 2 instead of -1 since some tensors might have an extra dimension.
        ret = torch.stack(lst, dim=2).refine_names(*new_names)
    return ret


@dataclass
class Beam:
    step: int
    accum_scores: FT
    tokens: LT  # Predicted tokens from last step.
    lstm_state: LstmStatesByLayers  # Last LSTM state.
    constants: BeamConstant
    # For bookkeeping.
    last_beam: Optional[Beam] = None
    beam_ids: Optional[LT] = None
    almt: Optional[FT] = None
    prev_att: Optional[FT] = None
    finished: BT = None

    def __post_init__(self):
        if self.finished is None:
            self.finished = torch.zeros_like(self.tokens).bool()

    @property
    def batch_size(self):
        return self.tokens.size('batch')

    @property
    def beam_size(self):
        return self.tokens.size('beam')

    def follow(self, finished: BT, accum_scores: FT, tokens: LT, lstm_state: LstmStatesByLayers, beam_ids: LT, almt: FT,
               prev_att: Optional[FT] = None) -> Beam:
        return Beam(self.step + 1, accum_scores, tokens,
                    lstm_state, self.constants,
                    last_beam=self,
                    beam_ids=beam_ids,
                    almt=almt,
                    prev_att=prev_att,
                    finished=finished)

    def trace_back(self, *attr_names: str) -> Dict[str, torch.Tensor]:
        """Trace back some attribute by going backwards through the beam search procedure."""
        beam_i = get_named_range(self.beam_size, 'beam').expand_as(self.beam_ids)
        batch_i = get_named_range(self.batch_size, 'batch').expand_as(beam_i)
        beam = self
        ret = defaultdict(list)
        while beam.last_beam is not None:
            with NoName(beam.beam_ids, beam_i, batch_i):
                for attr_name in attr_names:
                    attr = getattr(beam, attr_name)
                    with NoName(attr):
                        ret[attr_name].append(attr[batch_i, beam_i])
                beam_i = beam.beam_ids[batch_i, beam_i]
            beam = beam.last_beam
        for attr_name in attr_names:
            # NOTE(j_luo) Reverse the list since we are going backwards.
            last_name = 'src_pos' if attr_name == 'almt' else None
            ret[attr_name] = _stack_beam(ret[attr_name][::-1],
                                         last_name=last_name)
        return ret

    def to_traceback(self) -> BeamTraceback:
        """Return a `BeamTraceback` object. Note that this is different from the `trace_back` method
        because this only assembles all time steps together in a block-like fashion, whereas `trace_back`
        actually needs to reconstruct the sequences."""
        beam = self
        scores = list()
        beam_ids = list()
        tokens = list()
        while beam.last_beam is not None:
            scores.insert(0, beam.accum_scores)
            beam_ids.insert(0, beam.beam_ids)
            tokens.insert(0, beam.tokens)
            beam = beam.last_beam

        def to_nda(tensor: torch.Tensor) -> NDA:
            return tensor.cpu().detach().numpy()

        scores = to_nda(_stack_beam(scores))
        beam_ids = to_nda(_stack_beam(beam_ids))
        tokens = to_nda(_stack_beam(tokens))
        return BeamTraceback(scores, beam_ids, tokens)


class BeamTraceback:

    def __init__(self, scores: NDA, beam_ids: NDA, values: NDA):
        self.scores = scores
        self.beam_ids = beam_ids
        self.values = values

    def visualize(self, batch_index: int, output_name: str):
        """Visualize the entire search procedure for the `batch_index`-th example, saved to `output_name`."""
        g = Digraph('G', engine="neato", filename=output_name, format='svg')

        g.attr(size='7')
        s = self.scores[batch_index]
        b = self.beam_ids[batch_index]
        v = self.values[batch_index]
        B, L = s.shape

        def get_node_name(step, beam_id):
            return f'{step + 1},{beam_id + 1},{v[beam_id, step]},{s[beam_id, step]:.3f}'

        for t in range(L):
            for bi in range(B):
                node_name = get_node_name(t, bi)
                pos = f'{3 * t},{(B - bi)}!'
                g.node(node_name, pos=pos)
        for t in range(1, L):
            for bi in range(B):
                in_node = get_node_name(t, bi)
                out_node = get_node_name(t - 1, b[bi, t])
                g.edge(out_node, in_node)
        g.render()


@dataclass
class Candidates:
    log_probs: FT
    state: LstmStatesByLayers  # The LSTM state that is mapped to `log_probs`.
    almt: FT
    att: FT


@dataclass
class Hypotheses:
    tokens: LT
    almt: FT
    scores: FT

    def translate(self, abc: Alphabet) -> Tuple[NDA, NDA, NDA]:
        beam_translate = handle_sequence_inputs(lambda token_ids: translate(token_ids, abc=abc))
        pred_lengths = list()
        preds = list()
        properly_ended = list()
        for tokens in self.tokens.cpu().numpy():
            p, l, e = zip(*beam_translate(tokens))
            preds.append(p)
            pred_lengths.append(l)
            properly_ended.append(e)
        preds = np.asarray(preds)
        pred_lengths = np.asarray(pred_lengths)
        properly_ended = np.asarray(properly_ended)
        return preds, pred_lengths, properly_ended


def get_beam_probs(scores: FT, duplicates: Optional[BT] = None):
    """Return normalized scores (approximated probabilities) for the entire beam."""
    if duplicates is not None:
        scores = scores.masked_fill(duplicates, float('-inf'))
    return (scores / g.concentration_scale).log_softmax(dim='beam').exp()


class LstmDecoder(nn.Module, BaseBeamSearcher):
    """A decoder that unrolls the LSTM decoding procedure by steps."""

    add_argument('input_feeding', default=False, dtype=bool, msg='Flag to use input feeding.')

    def __init__(self,
                 char_emb: CharEmbedding,
                 cell: MultiLayerLSTMCell,
                 attn: GlobalAttention,
                 hidden: nn.Linear,
                 nc_residual: NormControlledResidual,
                 dropout: float = 0.0):
        super().__init__()
        self.char_emb = char_emb
        self.cell = cell
        self.attn = attn
        self.hidden = hidden
        self.nc_residual = nc_residual
        self.drop = nn.Dropout(dropout)

    @classmethod
    def from_params(cls,
                    dec_params: DecParams,
                    embedding: Optional[CharEmbedding] = None) -> LstmDecoder:
        emb_params = dec_params.emb_params
        lstm_params = dec_params.lstm_params
        if emb_params is None and embedding is None:
            raise ValueError('Must specify either `emb_params` or `embedding`.')

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

        return LstmDecoder(char_emb, cell, attn, hidden, nc_residual, lstm_params.dropout)

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
        prev_att = get_zeros(batch_size, g.hidden_size) if g.input_feeding else None
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
                state, log_prob, almt_distr, prev_att = self._forward_step(
                    input_, src_emb, state, src_outputs, mask_src,
                    lang_emb=lang_emb, prev_att=prev_att)
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
                      lang_emb: Optional[FT] = None,
                      prev_att: Optional[FT] = None) -> Tuple[FT, FT, FT, FT]:
        emb = self.char_emb(input_)
        if lang_emb is not None:
            emb = emb + lang_emb
        inp = torch.cat([emb, prev_att], dim=-1) if g.input_feeding else emb
        hid_rnn, next_state = self.cell(inp, state)  # hid_rnn has gone through dropout already.
        almt, ctx = self.attn.forward(hid_rnn, src_states, mask_src)  # So has src_states.
        with NoName(hid_rnn, ctx):
            cat = torch.cat([hid_rnn, ctx], dim=-1)
        hid_cat = self.hidden(cat)
        hid_cat = self.drop(hid_cat)

        with NoName(src_emb, hid_cat, almt):
            ctx_emb = (src_emb * almt.t().unsqueeze(dim=-1)).sum(dim=0)
            hid_res = self.nc_residual(ctx_emb, hid_cat).rename('batch', 'hidden')

        logit = self.char_emb.project(hid_res)
        log_prob = logit.log_softmax(dim=-1).refine_names('batch', 'unit')

        return next_state, log_prob, almt, hid_res

    def is_finished(self, beam: Beam) -> BT:
        return beam.finished

    def get_next_candidates(self, beam: Beam) -> Candidates:
        nh = NameHelper()

        def collapse_beam(orig, is_lstm_state: bool = False):

            def wrapped(tensor):
                return nh.flatten(tensor, ['batch', 'beam'], 'BB').rename(BB='batch')

            if is_lstm_state:
                return orig.apply(wrapped)
            return wrapped(orig)

        prev_att = collapse_beam(beam.prev_att) if g.input_feeding else None
        state, log_probs, almt, att = self._forward_step(
            collapse_beam(beam.tokens),
            beam.constants.src_emb,
            collapse_beam(beam.lstm_state, is_lstm_state=True),
            beam.constants.src_outputs,
            beam.constants.src_paddings,
            lang_emb=beam.constants.lang_emb,
            prev_att=prev_att)

        def unflatten(orig, is_lstm_state: bool = False):
            def wrapped(tensor):
                return nh.unflatten(tensor.rename(batch='BB'), 'BB', ['batch', 'beam'])

            if is_lstm_state:
                return orig.apply(wrapped)
            return wrapped(orig)

        log_probs = unflatten(log_probs)
        state = unflatten(state, is_lstm_state=True)
        almt = unflatten(almt)
        att = unflatten(att)
        return Candidates(log_probs, state, almt, att)

    def get_next_beam(self, beam: Beam, cand: Candidates) -> Beam:
        nh = NameHelper()

        # Get the new scores. For finished hypotheses, we should keep adding EOT.
        placeholder = torch.full_like(cand.log_probs, -9999.9)
        placeholder[..., EOT_ID] = 0.0
        new_scores = torch.where(beam.finished.align_as(placeholder), placeholder, cand.log_probs)
        accum = new_scores + beam.accum_scores.align_as(cand.log_probs)
        lp = nh.flatten(accum, ['beam', 'unit'], 'BU')
        top_s, top_i = torch.topk(lp, beam.beam_size, dim='BU')
        num_units = accum.size('unit')
        beam_i = top_i // num_units
        tokens = top_i % num_units

        batch_i = get_named_range(beam.batch_size, 'batch')
        batch_i = batch_i.align_as(top_i)

        def retrieve(tensor, last_name: str = 'hidden') -> torch.Tensor:
            with NoName(tensor, batch_i, beam_i):
                ret = tensor[batch_i, beam_i]
            new_names = ('batch', 'beam')
            if last_name:
                new_names += (last_name, )
            return ret.refine_names(*new_names)

        next_scores = top_s.rename(BU='beam')
        next_tokens = tokens.rename(BU='beam')
        next_beam_ids = beam_i.rename(BU='beam')
        next_state = cand.state.apply(retrieve)
        next_almt = retrieve(cand.almt, last_name='tgt_pos')
        next_att = retrieve(cand.att, last_name='hidden') if g.input_feeding else None
        last_finished = retrieve(beam.finished, last_name=None)
        this_ended = next_tokens == EOT_ID
        reached_max = (beam.step + 1 == beam.constants.max_lengths)
        next_finished = last_finished | this_ended | reached_max
        next_beam = beam.follow(next_finished,
                                next_scores,
                                next_tokens,
                                next_state,
                                next_beam_ids,
                                next_almt,
                                prev_att=next_att)
        return next_beam

    def search(self,
               sot_id: int,
               src_emb: FT,
               src_outputs: FT,
               src_paddings: BT,
               src_lengths: LT,
               beam_size: int,
               lang_emb: Optional[FT] = None) -> Hypotheses:
        if beam_size <= 0:
            raise ValueError(f'`beam_size` must be positive.')

        batch_size = src_emb.size('batch')
        tokens = torch.full([batch_size, beam_size], sot_id, dtype=torch.long).to(
            src_emb.device).rename('batch', 'beam')
        accum_scores = torch.full_like(tokens, -9999.9).float()
        accum_scores[:, 0] = 0.0
        init_att = None
        if g.input_feeding:
            init_att = get_zeros(batch_size, beam_size, g.hidden_size).rename('batch', 'beam', 'hidden')
        lstm_state = LstmStatesByLayers.zero_state(
            self.cell.num_layers,
            batch_size,
            beam_size,
            self.attn.input_tgt_size,
            bidirectional=False,
            names=['batch', 'beam', 'hidden'])

        def expand_beam(orig, collapse: bool = True):
            if collapse:
                return torch.repeat_interleave(orig, beam_size, dim='batch')
            else:
                return duplicate(orig, 'batch', beam_size, 'beam')

        src_emb = expand_beam(src_emb)
        src_outputs = expand_beam(src_outputs)
        src_paddings = expand_beam(src_paddings)
        max_lengths = (src_lengths.float() * 1.5).long()
        max_lengths = expand_beam(max_lengths, collapse=False)
        constants = BeamConstant(src_emb, src_outputs, src_paddings,
                                 max_lengths,
                                 lang_emb=lang_emb)
        init_beam = Beam(0, accum_scores, tokens,
                         lstm_state, constants, prev_att=init_att)
        hyps = super().search(init_beam)
        return hyps

    def get_hypotheses(self, final_beam: Beam) -> Hypotheses:
        btb = final_beam.trace_back('tokens', 'almt')
        tokens = btb['tokens']
        almt = btb['almt']
        return Hypotheses(tokens, almt, final_beam.accum_scores)

import logging
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dev_misc import BT, FT, LT, get_zeros
from dev_misc.devlib.named_tensor import NameHelper, NoName
from sound_law.model.lstm_state import LstmStatesByLayers, LstmStateTuple
from sound_law.data.dataset import PAD_ID

LstmOutputsByLayers = Tuple[FT, LstmStatesByLayers]
LstmOutputTuple = Tuple[FT, LstmStateTuple]

# TODO(j_luo) Function signatures for these classes get too complicated, which makes inheritance annoying and prone to error. Refactoring is needed here.


class MultiLayerLSTMCell(nn.Module):
    """An LSTM cell with multiple layers."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)

        cells = [nn.LSTMCell(input_size, hidden_size)]
        for _ in range(self.num_layers - 1):
            cells.append(nn.LSTMCell(hidden_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, input_: FT, state: LstmStatesByLayers, state_direction: Optional[str] = None) -> LstmOutputsByLayers:
        assert state.num_layers == self.num_layers

        new_states = list()
        for i in range(self.num_layers):
            # Note that the last layer doesn't use dropout.
            input_ = self.drop(input_)
            with NoName(input_):
                new_state = self.cells[i](input_, state.get_layer(i, state_direction))
            new_states.append(new_state)
            input_ = new_state[0].refine_names('batch', ...)
        return input_, LstmStatesByLayers(new_states)

    def extra_repr(self):
        return '%d, %d, num_layers=%d' % (self.input_size, self.hidden_size, self.num_layers)


class SharedEmbedding(nn.Embedding):
    """Shared input and output embedding."""

    def project(self, h: FT) -> FT:
        return h @ self.weight.t()


class PhonoEmbedding(SharedEmbedding):

    def __init__(self, phono_feat_mat: LT, special_ids: Sequence[int], num_embeddings: int, embedding_dim: int, *args, **kwargs):
        num_phones, num_features = phono_feat_mat.shape
        if embedding_dim % num_features > 0:
            raise ValueError(
                f'Embedding size {embedding_dim} cannot be divided by number of phonological features {num_features}.')
        super().__init__(num_embeddings, embedding_dim // num_features, *args, **kwargs)

        self.register_buffer('pfm', phono_feat_mat)
        self.special_weight = nn.Parameter(torch.randn(num_phones, embedding_dim))  # NOTE(j_luo) Use the undivided dim.
        special_mask = torch.zeros(num_phones).bool()
        special_mask[special_ids] = True
        self.register_buffer('special_mask', special_mask)

    @property
    def real_weight(self) -> FT:
        emb = super().forward(self.pfm)
        emb = emb.refine_names(..., 'phono_emb')
        nh = NameHelper()
        emb = nh.flatten(emb, ['phono_feat', 'phono_emb'], 'emb')
        return torch.where(self.special_mask.view(-1, 1), self.special_weight, emb)

    def forward(self, input_: LT) -> FT:
        with NoName(self.real_weight, input_):
            return self.real_weight[input_]

    def project(self, h: FT) -> FT:
        return h @ self.real_weight.t()


class LstmCellWithEmbedding(nn.Module):
    """An LSTM cell on top of an embedding layer."""

    def __init__(self,
                 num_embeddings: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 embedding: Optional[nn.Module] = None,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):
        super().__init__()

        if embedding is not None:
            logging.info(f'Using a shared embedding for LSTM.')

        if phono_feat_mat is not None and embedding is None:
            self.embedding = PhonoEmbedding(phono_feat_mat, special_ids, num_embeddings, input_size)
        else:
            self.embedding = embedding or SharedEmbedding(num_embeddings, input_size)
        self.lstm = MultiLayerLSTMCell(input_size, hidden_size, num_layers, dropout=dropout)

    def embed(self, input_: LT) -> FT:
        return self.embedding(input_)

    def forward(self,
                input_: LT,
                init_state: Optional[LstmStatesByLayers] = None,
                state_direction: Optional[str] = None) -> LstmOutputsByLayers:
        emb = self.embed(input_)

        batch_size = input_.size('batch')
        init_state = init_state or LstmStateTuple.zero_state(self.lstm.num_layers,
                                                             batch_size,
                                                             self.lstm.hidden_size)
        output, state = self.lstm(emb, init_state, state_direction=state_direction)
        return output, state


class LstmDecoder(LstmCellWithEmbedding):
    """A decoder that unrolls the LSTM decoding procedure by steps."""

    def forward(self,
                sot_id: int,  # "start-of-token"
                init_state: LstmStatesByLayers,
                max_length: Optional[int] = None,
                target: Optional[LT] = None,
                init_state_direction: Optional[str] = None) -> FT:
        max_length = self._get_max_length(max_length, target)
        input_ = self._prepare_first_input(sot_id, init_state.batch_size, init_state.device)
        log_probs = list()
        state = init_state
        state_direction = init_state_direction
        for l in range(max_length):
            input_, state, log_prob = self._forward_step(
                l, input_, state, target=target, state_direction=state_direction)
            # NOTE(j_luo) Only the first state uses actual `init_state_direction`.
            state_direction = None
            log_probs.append(log_prob)

        return self._gather_log_probs(log_probs)

    def _gather_log_probs(self, log_probs: List[FT]) -> FT:
        with NoName(*log_probs):
            log_probs = torch.stack(log_probs, dim=0).refine_names('pos', 'batch', 'unit')
        return log_probs

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
                      step: int,
                      input_: FT,
                      state: LstmStatesByLayers,
                      target: Optional[LT] = None,
                      state_direction: Optional[str] = None) -> Tuple[FT, FT, FT]:
        """Do one step of decoding. Return the input and the state for the next step, as well as the log probs."""
        output, next_state = super().forward(input_, state, state_direction=state_direction)
        logit = self.embedding.project(output)
        log_prob = logit.log_softmax(dim=-1)

        if self.training:
            next_input = target[step]
        else:
            next_input = log_prob.max(dim=-1)[1]

        return next_input, next_state, log_prob


class GlobalAttention(nn.Module):

    def __init__(self,
                 input_src_size: int,
                 input_tgt_size: int,
                 dropout: float = 0.0):
        super(GlobalAttention, self).__init__()

        self.input_src_size = input_src_size
        self.input_tgt_size = input_tgt_size

        self.Wa = nn.Parameter(torch.Tensor(input_src_size, input_tgt_size))
        torch.nn.init.xavier_normal_(self.Wa)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                h_t: FT,
                h_s: FT,
                mask_src: BT) -> Tuple[FT, FT]:
        sl, bs, ds = h_s.size()
        dt = h_t.shape[-1]
        Wh_s = self.drop(h_s).reshape(sl * bs, -1).mm(self.Wa).view(sl, bs, -1)

        with NoName(h_t):
            scores = (Wh_s * h_t).sum(dim=-1)

        scores = torch.where(mask_src, scores, torch.full_like(scores, -9999.9))
        almt_distr = nn.functional.log_softmax(scores, dim=0).exp()  # sl x bs
        with NoName(almt_distr):
            ctx = (almt_distr.unsqueeze(dim=-1) * h_s).sum(dim=0)  # bs x d
        almt_distr = almt_distr.t()
        return almt_distr, ctx

    def extra_repr(self):
        return 'src=%d, tgt=%d' % (self.input_src_size, self.input_tgt_size)


class NormControlledResidual(nn.Module):

    def __init__(self, norms_or_ratios=None, multiplier=1.0, control_mode=None):
        super().__init__()

        assert control_mode in ['none', 'relative', 'absolute']

        self.control_mode = control_mode
        self.norms_or_ratios = None
        if self.control_mode in ['relative', 'absolute']:
            self.norms_or_ratios = norms_or_ratios
            if self.control_mode == 'relative':
                assert self.norms_or_ratios[0] == 1.0

        self.multiplier = multiplier

    def anneal_ratio(self):
        if self.control_mode == 'relative':
            new_ratios = [self.norms_or_ratios[0]]
            for r in self.norms_or_ratios[1:]:
                r = min(r * self.multiplier, 1.0)
                new_ratios.append(r)
            self.norms_or_ratios = new_ratios
            logging.debug('Ratios are now [%s]' % (', '.join(map(lambda f: '%.2f' % f, self.norms_or_ratios))))

    def forward(self, *inputs):
        if self.control_mode == 'none':
            output = sum(inputs)
        else:
            assert len(inputs) == len(self.norms_or_ratios)
            outs = list()
            if self.control_mode == 'absolute':
                for inp, norm in zip(inputs, self.norms_or_ratios):
                    if norm >= 0.0:  # NOTE(j_luo) a negative value means no control applied
                        outs.append(normalize(inp, dim=-1) * norm)
                    else:
                        outs.append(inp)
            else:
                outs.append(inputs[0])
                norm_base = inputs[0].norm(dim=-1, keepdim=True)
                for inp, ratio in zip(inputs[1:], self.norms_or_ratios[1:]):
                    if ratio >= 0.0:  # NOTE(j_luo) same here
                        norm_actual = inp.norm(dim=-1, keepdim=True)
                        max_norm = norm_base * ratio
                        too_big = norm_actual > max_norm
                        adjusted_norm = torch.where(too_big, max_norm, norm_actual)
                        outs.append(normalize(inp, dim=-1) * adjusted_norm)
                    else:
                        outs.append(inp)
            output = sum(outs)
        return output


class LstmDecoderWithAttention(LstmDecoder):

    def __init__(self,
                 num_embeddings: int,
                 input_size: int,
                 src_hidden_size: int,
                 tgt_hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 control_mode: str = 'relative',
                 embedding: Optional[nn.Module] = None,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None,
                 norms_or_ratios: Optional[Tuple[float]] = None):
        super().__init__(num_embeddings, input_size, tgt_hidden_size, num_layers,
                         dropout=dropout, embedding=embedding)
        self.src_hidden_size = src_hidden_size
        self.tgt_hidden_size = tgt_hidden_size

        self.attn = GlobalAttention(src_hidden_size, tgt_hidden_size)
        self.hidden = nn.Linear(src_hidden_size + tgt_hidden_size, tgt_hidden_size)

        self.nc_residual = NormControlledResidual(norms_or_ratios, control_mode=control_mode)

    def forward(self,
                sot_id: int,
                src_emb: FT,
                src_states: FT,
                mask_src: BT,
                max_length: Optional[int] = None,
                target: Optional[LT] = None,
                lang_emb: Optional[FT] = None) -> Tuple[FT, FT]:
        max_length = self._get_max_length(max_length, target)
        batch_size = mask_src.size('batch')
        input_ = self._prepare_first_input(sot_id, batch_size, mask_src.device)
        state = LstmStatesByLayers.zero_state(self.lstm.num_layers,
                                              batch_size,
                                              self.tgt_hidden_size,
                                              bidirectional=False)
        log_probs = list()
        almt_distrs = list()
        for l in range(max_length):
            input_, state, log_prob, almt_distr = self._forward_step(
                l, input_, src_emb, state, src_states, mask_src, target=target, lang_emb=lang_emb)
            log_probs.append(log_prob)
            almt_distrs.append(almt_distr)
        log_probs = self._gather_log_probs(log_probs)
        with NoName(*almt_distrs):
            almt_distrs = torch.stack(almt_distrs).rename('tgt_pos', 'batch', 'src_pos')
        return log_probs, almt_distrs

    def _forward_step(self,
                      step: int,
                      input_: FT,
                      src_emb: FT,
                      state: LstmStatesByLayers,
                      src_states: FT,
                      mask_src: BT,
                      target: Optional[LT] = None,
                      lang_emb: Optional[FT] = None) -> Tuple[FT, FT, FT, FT]:
        emb = self.embed(input_)
        if lang_emb is not None:
            emb = emb + lang_emb
        hid_rnn, next_state = self.lstm(emb, state)
        almt, ctx = self.attn.forward(hid_rnn, src_states, mask_src)
        cat = torch.cat([hid_rnn, ctx], dim=-1)
        hid_cat = self.hidden(cat)

        with NoName(src_emb, hid_cat, almt):
            ctx_emb = (src_emb * almt.t().unsqueeze(dim=-1)).sum(dim=0)
            hid_res = self.nc_residual(ctx_emb, hid_cat)

        logit = self.embedding.project(hid_res)
        log_prob = logit.log_softmax(dim=-1)

        if self.training:
            next_input = target[step]
        else:
            next_input = log_prob.max(dim=-1)[1]
        return next_input, next_state, log_prob, almt


class LstmEncoder(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 embedding: Optional[nn.Module] = None,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):
        super().__init__()
        if phono_feat_mat is not None and embedding is None:
            self.embedding = PhonoEmbedding(phono_feat_mat, special_ids, num_embeddings, input_size)
        else:
            self.embedding = embedding or SharedEmbedding(num_embeddings, input_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)

    def forward(self, input_: LT, lengths: LT) -> Tuple[FT, LstmOutputTuple]:
        emb = self.embedding(input_)
        with NoName(emb, lengths):
            packed_emb = pack_padded_sequence(emb, lengths, enforce_sorted=False)
            output, state = self.lstm(packed_emb)
            output = pad_packed_sequence(output)[0]
        return emb, output, LstmStateTuple(state, bidirectional=self.lstm.bidirectional)


class CnnEncoder(nn.Module):
    
    def __init__(self,
                 num_embeddings: int,
                 input_size: int,
                 hidden_size: int,
                 kernel_sizes: Tuple[int, ...] = (3, 5, 7),
                 stride: int = 1,
                 dropout: float = 0.0,
                 embedding: Optional[nn.Module] = None,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):
        super().__init__()
        assert input_size == hidden_size # the layers' dimensionalities rely on this assumption

        if phono_feat_mat is not None and embedding is None:
            self.embedding = PhonoEmbedding(phono_feat_mat, special_ids, num_embeddings, input_size)
        else:
            self.embedding = embedding or SharedEmbedding(num_embeddings, input_size)

        # we want all the convolutional layers to create outputs with the same length. It's easiest to calculate the padding to do so when the kernel sizes are odd, so for now we only support odd kernel sizes.
        assert all(map(lambda x: x%2 == 1, kernel_sizes))

        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            # we pad the layers so that the output for each layer is of length seq_length
            padding = int((kernel_size - 1) / 2)
            layer = nn.Conv1d(input_size, hidden_size,
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding)
            self.conv_layers.append(layer)

        self.dropout = nn.Dropout(dropout)

        # the input size of this output projection layer depends on the number of convolutional modules
        n_conv = len(kernel_sizes)
        self.W_output = nn.Linear(n_conv*hidden_size, 2*hidden_size)

    def forward(self, input_: LT, lengths: LT) -> Tuple[FT, LstmOutputTuple]:
        # input_: seq_length x batch_size
        # note that input_size == hidden_size
        # define n_conv as the number of parallel convolutional layers
        emb = self.embedding(input_) # seq_length x batch_size x input_size
        
        with NoName(emb, lengths):
            reshaped_emb = emb.permute(1, 2, 0) # reshape to batch_size x input_size x seq_length for CNN input
            conv_outputs = [self.dropout(F.relu(conv(reshaped_emb))) for conv in self.conv_layers]
            # each conv layer's output is batch_size x hidden_size x seq_length

            # stack the CNN outputs on the hidden_size dimension
            x = torch.cat(conv_outputs, dim=1) # batch_size x n_conv*hidden_size x seq_length
            x = x.permute(2, 0, 1) # seq_length x batch_size x n_conv*hidden_size

            # project the concatenated convolutional layer outputs into 2*hidden_size dimensions so that `output` looks as though it were the states of a bidirectional lstm
            output = self.W_output(x) # seq_length x batch_size x 2*hidden_size
            # we don't try to reconstruct the state, so we just pass (None, None)
        return emb, output, (None, None)


class LanguageEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, unseen_idx: Optional[int] = None, mode: str = 'random', **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.unseen_idx = unseen_idx
        assert mode in ['random', 'mean']
        self.mode = mode

    def forward(self, index: int) -> FT:
        if index == self.unseen_idx:
            if self.mode == 'random':
                return self.weight[index]
            elif self.mode == 'mean':
                return (self.weight.sum(dim=0) - self.weight[index]) / (self.num_embeddings - 1)
        else:
            return self.weight[index]

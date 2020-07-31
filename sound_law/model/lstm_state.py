
'''
Just a wrapper of LSTM states: a list of (h, c) tuples
'''
from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Optional, Tuple

import torch

from dev_misc import FT, get_zeros

_StateTuple = Tuple[FT, FT]


class LstmState(ABC):

    @abstractmethod
    def __init__(self): ...


class LstmStateTuple(LstmState):
    """A class to represent a typical LSTM state tuple (h, c) following PyTorch's original format:
    `num_layers * num_directions, batch, hidden_size`.
    """

    def __init__(self, state: _StateTuple, bidirectional: bool = False):
        self._h, self._c = state
        self.num_layers = self._h.size(0) // (2 if bidirectional else 1)
        self.bidirectional = bidirectional
        self.batch_size = self._h.size(1)
        self.hidden_size = self._h.size(2)

    @classmethod
    def zero_state(cls,
                   num_layers: int,
                   batch_size: int,
                   hidden_size: int,
                   bidirectional: bool = False) -> LstmStateTuple:
        shape = (num_layers * (1 + bidirectional), batch_size, hidden_size)
        h = get_zeros(*shape)
        c = get_zeros(*shape)
        return LstmStateTuple((h, c), bidirectional=bidirectional)

    def to_layers(self) -> LstmStatesByLayers:
        hs = self._h.unbind(dim=0)
        cs = self._c.unbind(dim=0)
        if self.bidirectional:
            forward_states = list(zip(hs[::2], cs[::2]))
            backward_states = list(zip(hs[1::2], cs[1::2]))
        else:
            forward_states = list(zip(hs, cs))
            backward_states = None
        return LstmStatesByLayers(forward_states, backward_states)

    def to_hc_tuple(self) -> _StateTuple:
        return self._h, self._c


_StatesByLayers = List[_StateTuple]


class LstmStatesByLayers(LstmState):
    """A class to represent a typical LSTM state tuple (h, c).

    h and c are stored layer by layer.
    """

    def __init__(self,
                 forward_states: _StatesByLayers,
                 backward_states: Optional[_StatesByLayers] = None):

        def get_hs_and_cs(states: _StatesByLayers) -> _StateTuple:
            hs = [h for h, c in states]
            cs = [c for h, c in states]
            return hs, cs

        self._f_hs, self._f_cs = get_hs_and_cs(forward_states)
        self.num_layers = len(forward_states)
        self.bidirectional = backward_states is not None
        if self.bidirectional:
            self._b_hs, self._b_cs = get_hs_and_cs(backward_states)

    def get_layer(self, layer_id: int, direction: Optional[str] = None) -> _StateTuple:
        if self.bidirectional:
            assert direction in ['forward', 'backward', 'sum']
        else:
            assert direction is None

        if direction == 'sum':
            return self._f_hs[layer_id] + self._b_hs[layer_id], self._f_cs[layer_id] + self._b_cs[layer_id]

        if self.bidirectional and direction == 'backward':
            hs = self._b_hs
            cs = self._b_cs
        else:
            hs = self._f_hs
            cs = self._f_cs
        return hs[layer_id], cs[layer_id]

    @property
    def batch_size(self) -> int:
        return self._f_hs[0].size(0)

    @property
    def device(self) -> torch.device:
        return self._f_hs[0].device

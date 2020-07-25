
'''
Just a wrapper of LSTM states: a list of (h, c) tuples
'''
from __future__ import annotations

from typing import List, Tuple

from dev_misc import FT, get_zeros

StateTuple = Tuple[FT, FT]
StateTuples = List[Tuple[FT, FT]]


class LstmState:
    ... # FIXME(j_luo) fill in this: should this be a list of states (by layer) or ?
    ... # FIXME(j_luo) fill in this: methods to go from original format to new format.

    def __init__(self, states: StateTuples):
        self.states = states

    @classmethod
    def zero_state(cls, num_layers: int, shape: Tuple[int, ...]) -> LstmState:
        states = list()
        for _ in range(num_layers):
            h = get_zeros(*shape)
            c = get_zeros(*shape)
            states.append((h, c))
        return LstmState(states)


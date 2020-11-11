from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import (ClassVar, Dict, Iterator, List, NewType, Optional,
                    Sequence, Set, Tuple, Union)

import numpy as np

import sound_law.data.data_loader as dl
import sound_law.rl.action as a
from dev_misc import BT, FT, LT, NDA, g, get_tensor
from dev_misc.devlib import pad_to_dense
from dev_misc.utils import (Singleton, cached_property,
                            is_main_process_and_thread)
from editdistance import eval_batch
from sound_law.data.alphabet import PAD_ID

from .mcts_fast import \
    PyTreeNode  # pylint: disable=no-name-in-module # FIXME(j_luo) move tree node to another pyx file?


class VocabStateSpace:
    """This is the factory class for creating VocabState."""

    def get_state(self,
                  seqs: Optional[dl.PaddedUnitSeqs] = None,
                  #   units: Optional[Sequence[Sequence[str]]] = None, # FIXME(j_luo) units are not used for now.
                  ids: Optional[NDA] = None,
                  lengths: Optional[NDA] = None,
                  end_state: Optional[VocabState] = None) -> VocabState:
        if seqs is not None:
            ids = seqs.ids.t()
            lengths = seqs.lengths.t()
        # NOTE(j_luo) Since memoryviews are used in the extension class, we have to make them contiguous.
        arr = np.ascontiguousarray(ids.cpu().numpy())
        lengths = np.ascontiguousarray(lengths.cpu().numpy())
        return VocabState(arr=arr, lengths=lengths, end_node=end_state)


class VocabState(PyTreeNode):
    """State representing the vocab. Use `VocabStateSpace` to create one instance."""

    @cached_property
    def tensor(self) -> LT:
        """Convert the state into a long tensor."""
        return get_tensor(self.vocab_array).t().contiguous().rename('pos', 'word')

    @property
    def q(self):
        return self.total_value / (1e-8 + self.action_count)


@dataclass
class TrEdge:
    """This represents one edge in the trajectories."""
    s0: VocabState
    a: a.SoundChangeAction
    s1: VocabState
    done: bool
    r: float
    mcts_pi: Optional[NDA]  # This stores the policy produced by MCTS.


class Trajectory:

    def __init__(self, init_state: VocabState, end_state: VocabState):
        self._states = [init_state]
        self._actions: List[a.SoundChangeAction] = list()
        self._rewards: List[float] = list()
        self._mcts_pis: List[NDA] = list()
        self._end_state = end_state
        self._done = False  # Whether the trajectory has reached the end state.

    def append(self,
               action: a.SoundChangeAction,
               state: VocabState,
               done: bool,
               reward: float,
               mcts_pi: Optional[NDA] = None):
        if self._done:
            raise RuntimeError(f'This trajectory has already ended.')

        self._actions.append(action)
        self._states.append(state)
        self._rewards.append(reward)
        if mcts_pi is not None:
            self._mcts_pis.append(mcts_pi)
        self._done = done

    @property
    def rewards(self) -> NDA:
        return np.asarray(self._rewards, dtype='float32')

    @property
    def done(self) -> bool:
        return self._done

    @property
    def latest_state(self) -> VocabState:
        return self._states[-1]

    def __len__(self):
        return len(self._actions)

    def __iter__(self) -> Iterator[TrEdge]:
        for i, (s0, a, r) in enumerate(zip(self._states, self._actions, self._rewards)):
            s1 = self._states[i + 1]
            done = False if i < len(self._actions) - 1 else self._done
            mcts_pi = self._mcts_pis[i] if self._mcts_pis else None
            yield TrEdge(s0, a, s1, done, r, mcts_pi=mcts_pi)

    def __repr__(self):
        out = list()
        for edge in self:
            out.append(f'({edge.a}; {edge.r:.3f})')
        out = ', '.join(out)
        if self._done:
            out += ' DONE'
        return out

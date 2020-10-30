from __future__ import annotations

from functools import lru_cache
from typing import (ClassVar, Dict, Iterator, List, NewType, Optional,
                    Sequence, Set, Tuple)

import numpy as np

import sound_law.data.data_loader as dl
import sound_law.rl.action as a
from dev_misc import BT, FT, LT, NDA
from dev_misc.utils import Singleton, cached_property
from sound_law.evaluate.edit_dist import ed_eval_batch


class Word:

    def __init__(self, units: List[str]):
        self.units = units
        self.key = ' '.join(units)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other: Word):
        return self.key == other.key

    def __repr__(self):
        return self.key


class VocabState:

    def __init__(self, units: Sequence[List[str]], ids: LT):
        self.units = units
        self.words = [Word(u) for u in units]
        self.ids = ids

    @classmethod
    def from_seqs(cls, seqs: dl.PaddedUnitSeqs) -> VocabState:
        return cls(seqs.units, seqs.ids)

    def __eq__(self, other: VocabState):
        return len(self.words) == len(other.words) and all(s == o for s, o in zip(self.words, other.words))

    @cached_property
    def hash_str(self) -> str:
        return '\n'.join([word.key for word in self.words])

    def __hash__(self):
        return id(self.hash_str)

    @lru_cache(maxsize=None)
    def dist_from(self, other: VocabState) -> float:
        units_1 = [word.units for word in self.words]
        units_2 = [word.units for word in other.words]
        return float(ed_eval_batch(units_1, units_2, 4).sum())


class Trajectory:

    def __init__(self, init_state: VocabState, end_state: VocabState):
        self._states = [init_state]
        self._actions: List[a.SoundChangeAction] = list()
        self._rewards: List[float] = list()
        self._action_masks: List[BT] = list()
        self._end_state = end_state
        self._done = False  # Whether the trajectory has reached the end state.

    def append(self, action: a.SoundChangeAction, state: VocabState, done: bool, reward: float, action_masks: BT):
        if self._done:
            raise RuntimeError(f'This trajectory has already ended.')

        self._actions.append(action)
        self._states.append(state)
        self._rewards.append(reward)
        self._action_masks.append(action_masks)
        self._done = done

    @property
    def rewards(self) -> NDA:
        return np.asarray(self._rewards)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def latest_state(self) -> VocabState:
        return self._states[-1]

    def __len__(self):
        return len(self._actions)

    def __iter__(self) -> Iterator[Tuple[VocabState, a.SoundChangeAction, VocabState, FT, BT]]:
        for i, (s0, a, r, am) in enumerate(zip(self._states, self._actions, self._rewards, self._action_masks)):
            s1 = self._states[i + 1]
            yield s0, a, s1, r, am

    def __repr__(self):
        out = list()
        for s0, a, s1, r, am in self:
            out.append(f'({a}; {r:.3f})')
        out = ', '.join(out)
        if self._done:
            out += ' DONE'
        return out

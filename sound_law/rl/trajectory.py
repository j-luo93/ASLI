from __future__ import annotations

from functools import lru_cache
from typing import (ClassVar, Dict, Iterator, List, NewType, Optional,
                    Sequence, Set, Tuple, Union)

import numpy as np

import sound_law.data.data_loader as dl
import sound_law.rl.action as a
from dev_misc import BT, FT, LT, NDA, g
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


SKey = NewType('SKey', str)  # State key.


class VocabStateSpace(Singleton):
    """The space of all vocab states. This handles the creation of vocab states."""

    _states: ClassVar[SKey, VocabState] = dict()

    def get_state(self, *,
                  seqs: Optional[dl.PaddedUnitSeqs] = None,
                  words: Optional[List[Word]] = None,
                  ids: Optional[Union[NDA, LT]] = None) -> VocabState:
        if seqs is not None:
            words = [Word(u) for u in seqs.units]
            ids = seqs.ids
            # NOTE(j_luo) For MCTS, we use numpy arrays for ids.
            if g.use_mcts:
                ids = np.ascontiguousarray(ids.cpu().numpy())
        s_key = '\n'.join([word.key for word in words])
        if s_key not in self._states:
            obj = VocabState(len(self._states), s_key, words, ids)
            self._states[s_key] = obj
        return self._states[s_key]


class VocabState:

    def __init__(self, s_id: int, s_key: SKey, words: List[Word], ids: Union[NDA, LT]):
        """This should not be directly called. Use VocabStateSpace to call `get_state` instead."""
        self.s_id = s_id  # The unique id for this state.
        self.s_key = s_key  # The unique string (key) for this state.
        self.words = words
        self.ids = ids

    def __eq__(self, other: VocabState):
        return self.s_id == other.s_id
        # return len(self.words) == len(other.words) and all(s == o for s, o in zip(self.words, other.words))

    def __hash__(self):
        return self.s_id

    # @lru_cache(maxsize=None)
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
        return np.asarray(self._rewards, dtype='float32')

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

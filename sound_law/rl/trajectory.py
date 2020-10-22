from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

from dev_misc import FT, LT
import sound_law.data.data_loader as dl

from .action import SoundChangeAction


class VocabState:

    def __init__(self, units: Sequence[List[str]], ids: LT):
        self.units = units
        self.ids = ids

    @classmethod
    def from_seqs(cls, seqs: dl.PaddedUnitSeqs) -> VocabState:
        return cls(seqs.units, seqs.ids)

    def __eq__(self, other: VocabState):
        return len(self.units) == len(other.units) and all(s == o for s, o in zip(self.units, other.units))


class Trajectory:

    def __init__(self, init_state: VocabState, end_state: VocabState):
        self._states = [init_state]
        self._actions: List[SoundChangeAction] = list()
        self._rewards: List[float] = list()
        self._end_state = end_state
        self._done = False  # Whether the trajectory has reached the end state.

    def append(self, action: SoundChangeAction, state: VocabState, done: bool, reward: float):
        if self._done:
            raise RuntimeError(f'This trajectory has already ended.')

        self._actions.append(action)
        self._states.append(state)
        self._rewards.append(reward)
        self._done = done

    @property
    def done(self) -> bool:
        return self._done

    @property
    def latest_state(self) -> VocabState:
        return self._states[-1]

    def __len__(self):
        return len(self._actions)

    def __iter__(self) -> Iterator[Tuple[VocabState, SoundChangeAction, VocabState, FT]]:
        for i, (s0, a, r) in enumerate(zip(self._states, self._actions, self._rewards)):
            s1 = self._states[i + 1]
            yield s0, a, s1, r

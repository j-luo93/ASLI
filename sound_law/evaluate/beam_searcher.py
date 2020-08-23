from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from dev_misc import BT, LT

B = TypeVar('Beam')
C = TypeVar('Candidates')


class BaseBeamCearcher(ABC):

    def __init__(self, batch_size: int, beam_size: int):
        self._batch_size = batch_size
        self._beam_size = beam_size

    def search(self, init_beam: B, max_lengths: LT) -> B:
        beam = init_beam
        while True:
            finished = self.is_finished(beam, max_lengths)
            if finished.all():
                break

            candidates = self.get_next_candidates(beam)
            beam = self.get_next_beam(beam, candidates)
        return beam

    @abstractmethod
    def is_finished(self, beam: B, max_lengths: LT) -> BT: ...

    @abstractmethod
    def get_next_candidates(self, beam: B) -> C: ...

    @abstractmethod
    def get_next_beam(self, beam: B, candidates: C) -> B: ...

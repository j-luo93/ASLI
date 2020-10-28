"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Iterator, List, Set

import sound_law.rl.trajectory as tr
from sound_law.data.alphabet import Alphabet


class SoundChangeActionSpace:
    """The action space, i.e., the space of all sound changes."""

    def __init__(self, abc: Alphabet):
        self.abc = abc
        self._actions: List[SoundChangeAction] = list()
        units = [u for u in self.abc if u not in self.abc.special_units]
        for u1, u2 in product(units, repeat=2):
            if u1 != u2:
                action = SoundChangeAction(len(self._actions), u1, u2, abc[u1], abc[u2])
                self._actions.append(action)
        logging.info(f'Number of actions in action space: {len(self._actions)}.')

    def __len__(self):
        return len(self._actions)

    def get_action(self, idx: int) -> SoundChangeAction:
        return self._actions[idx]

    def __iter__(self) -> Iterator[SoundChangeAction]:
        yield from self._actions

    @lru_cache(maxsize=100000)
    def get_permissible_actions(self, state: tr.VocabState) -> Set[SoundChangeAction]:
        ret = set()
        for word in state.words:
            action_set = self._get_permissible_word_actions(word)
            ret.update(action_set)
        return ret

    @lru_cache(maxsize=10000)
    def _get_permissible_word_actions(self, word: tr.Word) -> Set[SoundChangeAction]:
        units = set(word.units)
        return set(action for action in self._actions if action.before in units)


@dataclass
class SoundChangeAction:
    """One sound change rule."""

    action_id: int
    before: str
    after: str
    before_id: int
    after_id: int

    def __str__(self):
        return f'{self.before} > {self.after}'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: SoundChangeAction):
        return str(self) == str(other)

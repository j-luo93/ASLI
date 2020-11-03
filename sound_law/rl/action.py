"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Iterator, List, Set, Union

import sound_law.rl.trajectory as tr
from dev_misc import BT, add_argument, g, get_tensor, get_zeros
from dev_misc.utils import Singleton
from sound_law.data.alphabet import Alphabet


class SoundChangeActionSpace(Singleton):
    """The action space, i.e., the space of all sound changes."""

    add_argument('factorize_actions', dtype=bool, default=False, msg='Flag to factorize the action space.')

    def __init__(self, abc: Alphabet):
        self.abc = abc
        self._actions: List[SoundChangeAction] = list()
        units = [u for u in self.abc if u not in self.abc.special_units]
        for u1, u2 in product(units, repeat=2):
            if u1 != u2:
                action = SoundChangeAction(len(self._actions), u1, u2, abc[u1], abc[u2])
                self._actions.append(action)
        logging.info(f'Number of actions in action space: {len(self._actions)}.')

        if g.factorize_actions:
            a2b = list()
            a2a = list()
            for action in self._actions:
                a2b.append(action.before_id)
                a2a.append(action.after_id)
            self.action2before = get_tensor(a2b)
            self.action2after = get_tensor(a2a)

    def __len__(self):
        return len(self._actions)

    def get_action(self, idx: int) -> SoundChangeAction:
        return self._actions[idx]

    def __iter__(self) -> Iterator[SoundChangeAction]:
        yield from self._actions

    def get_permissible_actions(self, state: tr.VocabState, ret_tensor: bool = False) -> Union[SoundChangeAction, BT]:
        actions = self._get_permissible_actions(state)
        if ret_tensor:
            action_ids = [action.action_id for action in actions]
            action_masks = get_zeros(len(self._actions)).bool()
            action_masks[action_ids] = True
            return action_masks
        return actions

    @lru_cache(maxsize=100000)
    def _get_permissible_actions(self, state: tr.VocabState) -> Set[SoundChangeAction]:
        ret = set()
        for word in state.words:
            action_set = self._get_permissible_word_actions(word)
            ret.update(action_set)
        if not ret:
            raise RuntimeError(f'No permissible action for this state.')
        return ret

    @lru_cache(maxsize=10000)
    def _get_permissible_word_actions(self, word: tr.Word) -> Set[SoundChangeAction]:
        units = set(word.units)
        ret = set(action for action in self._actions if action.before in units)
        if not ret:
            raise RuntimeError(f'No permissible action for this word.')
        return ret


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

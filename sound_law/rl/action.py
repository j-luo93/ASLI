"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Iterator, List, Set, Union

import numpy as np

import sound_law.rl.trajectory as tr
from dev_misc import BT, add_argument, g, get_tensor, get_zeros
from dev_misc.utils import Singleton
from sound_law.data.alphabet import Alphabet

from .mcts_fast import PyAction, PyActionSpace  # pylint: disable=no-name-in-module

from typing import ClassVar

class SoundChangeAction(PyAction):
    """One sound change rule."""
    abc: ClassVar[Alphabet] = None

    # FIXME(j_luo) no unit so far.
    # action_id: int
    # before: str
    # after: str
    # before_id: int
    # after_id: int

    def __str__(self):
        if self.abc is not None:
            before = self.abc[self.before_id]
            after = self.abc[self.after_id]
            return f'{before} > {after}'
        return f'{self.before_id} > {self.after_id}'


class SoundChangeActionSpace(PyActionSpace):
    """The action space, i.e., the space of all sound changes."""
    action_cls = SoundChangeAction

    add_argument('factorize_actions', dtype=bool, default=False, msg='Flag to factorize the action space.')

    def __init__(self, abc: Alphabet):
        super().__init__()
        # Set class variable for `SoundChangeAction` here.
        self.abc = SoundChangeAction.abc = abc
        units = [u for u in self.abc if u not in self.abc.special_units]
        for u1, u2 in product(units, repeat=2):
            if u1 != u2:
                self.register_action(abc[u1], abc[u2])
        logging.info(f'Number of actions in action space: {len(self)}.')

        if g.factorize_actions:
            a2b = list()
            a2a = list()
            for action in self:
                a2b.append(action.before_id)
                a2a.append(action.after_id)
            self.action2before = get_tensor(a2b)
            self.action2after = get_tensor(a2a)

"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import ClassVar, Iterator, List, Set, Union

import numpy as np

import sound_law.rl.trajectory as tr
from dev_misc import BT, add_argument, g, get_tensor, get_zeros
from dev_misc.utils import Singleton
from sound_law.data.alphabet import Alphabet

from .mcts_fast import (PyAction,  # pylint: disable=no-name-in-module
                        PyActionSpace)


class SoundChangeAction(PyAction):
    """One sound change rule."""
    abc: ClassVar[Alphabet] = None

    # TODO(j_luo) no unit so far.
    # action_id: int
    # before: str
    # after: str
    # before_id: int
    # after_id: int

    def __repr__(self):
        if self.abc is not None:
            before = self.abc[self.before_id]
            after = self.abc[self.after_id]
            prefix = f'{self.abc[self.pre_id]} + ' if self.pre_id != -1 else ''
            return f'{prefix}{before} > {prefix}{after}'
        prefix = f'{self.pre_id} + ' if self.pre_id != -1 else ''
        return f'{prefix}{self.before_id} > {prefix}{self.after_id}'


class SoundChangeActionSpace(PyActionSpace):
    """The action space, i.e., the space of all sound changes."""
    action_cls = SoundChangeAction

    add_argument('factorize_actions', dtype=bool, default=False, msg='Flag to factorize the action space.')
    add_argument('max_dist', dtype=int, default=3, msg='Maximum distance to draw an edge between two characters.')

    def __init__(self, abc: Alphabet):
        super().__init__()
        # Set class variable for `SoundChangeAction` here.
        self.abc = SoundChangeAction.abc = abc
        units = [u for u in self.abc if u not in self.abc.special_units]
        for u1, u2 in product(units, repeat=2):
            if u1 != u2:
                id1 = abc[u1]
                id2 = abc[u2]
                if not g.use_mcts or abc.dist_mat[id1, id2] <= g.max_dist:
                    self.register_action(id1, id2)
                    if g.use_conditional:
                        for u in units:
                            self.register_action(id1, id2, abc[u])
        logging.info(f'Number of actions in action space: {len(self)}.')

        if g.factorize_actions:

            def gather(attr: str):
                ret = list()
                for action in self:
                    ret.append(getattr(action, attr))
                return get_tensor(ret)

            self.action2before = gather('before_id')
            self.action2after = gather('after_id')
            if g.use_conditional:
                self.action2pre = gather('pre_id')

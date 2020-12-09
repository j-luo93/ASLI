"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

from dev_misc.utils import pbar
from typing import Dict
import torch
from collections import defaultdict
import pickle
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

from .mcts.mcts_fast import PyAction, PyActionSpace, PyWordSpace, PySiteSpace  # pylint: disable=no-name-in-module
# from .mcts_fast import (PyAction,  # pylint: disable=no-name-in-module
#                         PyActionSpace)


class SoundChangeAction(PyAction):
    """One sound change rule."""
    abc: ClassVar[Alphabet] = None

    def __repr__(self):
        if self.action_id == 0:
            return 'STOP'

        def get_cond(cond):
            if self.abc is None:
                ret = ' + '.join(map(str, cond))
            else:
                ret = ' + '.join(map(str, [self.abc[i] for i in cond]))  # pylint: disable=unsubscriptable-object
            if ret:
                ret = f'({ret})'
            return ret

        pre = get_cond(self.pre_cond)
        if pre:
            pre = f'{pre} + '
        post = get_cond(self.post_cond)
        if post:
            post = f' + {post}'

        before = str(
            self.before_id) if self.abc is None else self.abc[self.before_id]  # pylint: disable=unsubscriptable-object
        after = str(
            self.after_id) if self.abc is None else self.abc[self.after_id]  # pylint: disable=unsubscriptable-object

        return f'{pre}{before}{post} > {after}'


class SoundChangeActionSpace(PyActionSpace):
    """The action space, i.e., the space of all sound changes."""

    action_cls = SoundChangeAction

    add_argument('factorize_actions', dtype=bool, default=False, msg='Flag to factorize the action space.')
    add_argument('ngram_path', dtype='path', msg='Path to the ngram list.')

    def __init__(self, py_ss: PySiteSpace, py_ws: PyWordSpace, num_threads: int, abc: Alphabet):
        super().__init__()
        # # Set class variable for `SoundChangeAction` here.
        self.abc = SoundChangeAction.abc = abc

        # Register unconditional actions first.
        units = [u for u in self.abc if u not in self.abc.special_units]
        possible_path: Dict[str, List[str]] = defaultdict(list)

        def register_uncondional_action(u1: str, u2: str):
            id1 = abc[u1]
            id2 = abc[u2]
            self.register_edge(id1, id2)
            possible_path[id1].append(id2)

        if g.use_mcts:
            for u1, u2 in abc.edges:
                register_uncondional_action(u1, u2)
        else:
            for u1, u2 in product(units, repeat=2):
                if u1 != u2:
                    register_uncondional_action(u1, u2)

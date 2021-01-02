"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import ClassVar, Dict, Iterator, List, Set, Union

import numpy as np
import torch

import sound_law.rl.trajectory as tr
from dev_misc import BT, add_argument, g, get_tensor, get_zeros
from dev_misc.utils import Singleton, pbar
from sound_law.data.alphabet import (ANY_ID, EMP, EMP_ID, EOT_ID, SOT_ID,
                                     Alphabet)

# pylint: disable=no-name-in-module
from .mcts_cpp.mcts_cpp import (PyAction, PyActionSpace, PySiteSpace, PyStop,
                                PyWordSpace)

# pylint: enable=no-name-in-module


class SoundChangeAction(PyAction):
    """One sound change rule."""
    abc: ClassVar[Alphabet] = None

    def __repr__(self):
        if self.action_id == PyStop:
            return 'STOP'

        def get_str(idx: int):
            if idx in [SOT_ID, EOT_ID]:
                return '#'
            elif idx == ANY_ID:
                return '.'
            elif idx == EMP_ID:
                return 'Ø'
            return self.abc[idx]  # pylint: disable=unsubscriptable-object

        def get_cond(cond):
            if self.abc is None:
                ret = ' + '.join(map(str, cond))
            else:
                ret = ' + '.join(map(str, [get_str(i) for i in cond]))
            if ret:
                ret = f'({ret})'
            return ret

        pre = get_cond(self.pre_cond)
        if pre:
            pre = f'{pre} + '
        post = get_cond(self.post_cond)
        if post:
            post = f' + {post}'

        before = str(self.before_id) if self.abc is None else get_str(self.before_id)
        after = str(self.after_id) if self.abc is None else get_str(self.after_id)

        return f'{pre}{before}{post} > {after}'


class SoundChangeActionSpace(PyActionSpace):
    """The action space, i.e., the space of all sound changes."""

    action_cls = SoundChangeAction

    add_argument('factorize_actions', dtype=bool, default=False, msg='Flag to factorize the action space.')
    add_argument('ngram_path', dtype='path', msg='Path to the ngram list.')

    def __init__(self, py_ss: PySiteSpace, py_ws: PyWordSpace, dist_threshold: float, site_threshold: int, abc: Alphabet):
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
            for u in units:
                register_uncondional_action(u, EMP)
        else:
            for u1, u2 in product(units, repeat=2):
                if u1 != u2:
                    register_uncondional_action(u1, u2)

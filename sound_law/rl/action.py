"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

from typing import Dict
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

from .mcts_fast import (PyAction,  # pylint: disable=no-name-in-module
                        PyActionSpace)


class SoundChangeAction(PyAction):
    """One sound change rule."""
    abc: ClassVar[Alphabet] = None

    def __repr__(self):

        def get_cond(cond):
            if self.abc is None:
                ret = ' + '.join(map(str, cond))
            else:
                ret = ' + '.join(map(str, [self.abc[i] for i in cond]))
            if ret:
                ret = f'({ret})'
            return ret

        pre = get_cond(self.pre_cond)
        if pre:
            pre = f'{pre} + '
        post = get_cond(self.post_cond)
        if post:
            post = f' + {post}'

        before = str(self.before_id) if self.abc is None else self.abc[self.before_id]
        after = str(self.after_id) if self.abc is None else self.abc[self.after_id]

        return f'{pre}{before}{post} > {after}'


class SoundChangeActionSpace(PyActionSpace):
    """The action space, i.e., the space of all sound changes."""
    action_cls = SoundChangeAction

    add_argument('factorize_actions', dtype=bool, default=False, msg='Flag to factorize the action space.')
    add_argument('ngram_path', dtype='path', msg='Path to the ngram list.')

    def __init__(self, abc: Alphabet):
        super().__init__()
        # Set class variable for `SoundChangeAction` here.
        self.abc = SoundChangeAction.abc = abc
        units = [u for u in self.abc if u not in self.abc.special_units]
        possible_path: Dict[str, List[str]] = defaultdict(list)
        for u1, u2 in product(units, repeat=2):
            if u1 != u2:
                id1 = abc[u1]
                id2 = abc[u2]
                if not g.use_mcts or (u1, u2) in abc.edges:
                    self.register_action(id1, id2)
                    possible_path[id1].append(id2)
        if g.use_conditional:
            with open(g.ngram_path, 'rb') as fin:
                ngram = pickle.load(fin)
            from tqdm import tqdm
            for tup, atype in tqdm(ngram):
                pre_cond = post_cond = None
                if atype == 'pre':
                    pre_cond = [tup[0]]
                    seg = tup[1]
                elif atype == 'post':
                    post_cond = [tup[1]]
                    seg = tup[0]
                elif atype == 'd_pre':
                    pre_cond = [tup[0], tup[1]]
                    seg = tup[2]
                elif atype == 'd_post':
                    post_cond = [tup[1], tup[2]]
                    seg = tup[0]
                elif atype == 'pre_post':
                    pre_cond = [tup[0]]
                    post_cond = [tup[2]]
                    seg = tup[1]
                elif atype == 'd_pre_post':
                    pre_cond = [tup[0], tup[1]]
                    post_cond = [tup[3]]
                    seg = tup[2]
                elif atype == 'pre_d_post':
                    pre_cond = [tup[0]]
                    post_cond = [tup[2], tup[3]]
                    seg = tup[1]
                elif atype == 'd_pre_d_post':
                    pre_cond = [tup[0], tup[1]]
                    post_cond = [tup[3], tup[4]]
                    seg = tup[2]
                pre_cond = [abc[i] for i in pre_cond] if pre_cond else None
                post_cond = [abc[i] for i in post_cond] if post_cond else None
                id1 = abc[seg]
                for id2 in possible_path[id1]:
                    self.register_action(id1, id2, pre_cond=pre_cond, post_cond=post_cond)

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
                self.action2d_pre = gather('d_pre_id')
                self.action2post = gather('post_id')
                self.action2d_post = gather('d_post_id')

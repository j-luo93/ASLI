"""This file defines methods and classes useful for representing sound change rules in the form of actions.
"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import (ClassVar, Dict, Iterator, List, Optional, Sequence, Set,
                    Union)

import numpy as np
import torch

import sound_law.rl.trajectory as tr
from dev_misc import BT, add_argument, g, get_tensor, get_zeros
from dev_misc.utils import Singleton, pbar
from sound_law.data.alphabet import (ANY_ID, ANY_S_ID, ANY_UNS_ID, EMP, EMP_ID,
                                     EOT_ID, NULL_ID, SOT_ID, SYL_EOT_ID,
                                     Alphabet)

add_argument('factorize_actions', dtype=bool, default=False, msg='Flag to factorize the action space.')
add_argument('ngram_path', dtype='path', msg='Path to the ngram list.')


@dataclass(eq=True, frozen=True)
class SoundChangeAction:
    """One sound change rule."""

    before_id: int
    after_id: int
    pre_id: int
    d_pre_id: int
    post_id: int
    d_post_id: int
    special_type: Optional[str] = None

    abc: ClassVar[Alphabet] = None

    # def __hash__(self):
    #     return hash(f"{self.special_type}: {self.before_id} {self.after_id} {self.pre_id} {self.d_pre_id} {self.post_id} {self.d_post_id}")

    # def __eq__(self, other: SoundChangeAction):
    #     return self.before_id == other.before_id and self.after_id == other.after_id and self.pre_id== self.other_id

    @classmethod
    def from_str(cls, before: str, after: str,
                 pre: Optional[str] = None,
                 d_pre: Optional[str] = None,
                 post: Optional[str] = None,
                 d_post: Optional[str] = None,
                 special_type: Optional[str] = None) -> SoundChangeAction:
        if cls.abc is None:
            raise RuntimeError(f"No alphabet has been specified.")
        if d_pre is not None and pre is None:
            raise ValueError(f"`pre` must be present for `d_pre`.")
        if d_post is not None and post is None:
            raise ValueError(f"`post` must be present for `d_post`.")

        def to_int(unit: Union[None, str], before_or_after: str) -> int:
            if unit in ["empty", 'Ø']:
                return EMP_ID
            if unit == '.':
                return ANY_ID
            if unit == '.{+}':
                return ANY_S_ID
            if unit == '.{-}':
                return ANY_UNS_ID
            if unit == "#":
                return SOT_ID if before_or_after == 'b' else EOT_ID
            if unit == '##':
                return SYL_EOT_ID
            if unit is None:
                return NULL_ID
            return cls.abc[unit]  # pylint: disable=unsubscriptable-object

        return cls(cls.abc[before], to_int(after, 'a'),  # pylint: disable=unsubscriptable-object
                   to_int(pre, 'b'), to_int(d_pre, 'b'),
                   to_int(post, 'a'), to_int(d_post, 'a'),
                   special_type=special_type)

    def __repr__(self):
        # if self.action_id == PyStop:
        if self.before_id == NULL_ID:
            return 'STOP'

        def get_str(idx: int):
            if idx in [SOT_ID, EOT_ID]:
                return '#'
            elif idx == ANY_ID:
                return '.'
            elif idx == EMP_ID:
                return 'Ø'
            elif idx == ANY_S_ID:
                return '.{+}'
            elif idx == ANY_UNS_ID:
                return '.{-}'
            return self.abc[idx]  # pylint: disable=unsubscriptable-object

        def get_cond(cond):
            if self.abc is None:
                ret = ' + '.join(map(str, cond))
            else:
                ret = ' + '.join(map(str, [get_str(i) for i in cond]))
            if ret:
                ret = f'({ret})'
            return ret

        pre = get_cond([idx for idx in [self.d_pre_id, self.pre_id] if idx != NULL_ID])
        if pre:
            pre = f'{pre} + '
        post = get_cond([idx for idx in [self.post_id, self.d_post_id] if idx != NULL_ID])
        if post:
            post = f' + {post}'

        before = str(self.before_id) if self.abc is None else get_str(self.before_id)
        after = str(self.after_id) if self.abc is None else get_str(self.after_id)

        special = '' if self.special_type is None else (self.special_type + ': ')
        return f'{special}{pre}{before}{post} > {after}'


# class SoundChangeActionSpace(PyActionSpace):
#     """The action space, i.e., the space of all sound changes."""

#     action_cls = SoundChangeAction

#     add_argument('factorize_actions', dtype=bool, default=False, msg='Flag to factorize the action space.')
#     add_argument('ngram_path', dtype='path', msg='Path to the ngram list.')

#     def __init__(self, py_ss: PySiteSpace, py_ws: PyWordSpace, dist_threshold: float, site_threshold: int, abc: Alphabet):
#         super().__init__()
#         # # Set class variable for `SoundChangeAction` here.
#         self.abc = SoundChangeAction.abc = abc

#         # Register unconditional actions first.
#         units = [u for u in self.abc if u not in self.abc.special_units]

#         def register_uncondional_action(u1: str, u2: str, cl: bool = False, gb: bool = False):
#             id1 = abc[u1]
#             id2 = abc[u2]
#             if cl:
#                 self.register_cl_map(id1, id2)
#             elif gb:
#                 if u1.startswith('i'):
#                     self.register_gbj(id1, id2)
#                 else:
#                     assert u1.startswith('u')
#                     self.register_gbw(id1, id2)
#             else:
#                 self.register_edge(id1, id2)

#         if g.use_mcts:
#             for u1, u2 in abc.edges:
#                 register_uncondional_action(u1, u2)
#             for u in units:
#                 register_uncondional_action(u, EMP)
#             for u1, u2 in abc.cl_map.items():
#                 register_uncondional_action(u1, u2, cl=True)
#             for u1, u2 in abc.gb_map.items():
#                 register_uncondional_action(u1, u2, gb=True)
#         else:
#             for u1, u2 in product(units, repeat=2):
#                 if u1 != u2:
#                     register_uncondional_action(u1, u2)

#         self.set_vowel_info(abc.vowel_mask, abc.vowel_base, abc.vowel_stress, abc.stressed_vowel, abc.unstressed_vowel)
#         self.set_glide_info(abc['j'], abc['w'])

#     def apply_action(self, unit_seq: Sequence[str], action: SoundChangeAction) -> List[str]:
#         id_seq = [self.abc[u] for u in unit_seq]
#         new_id_seq = super().apply_action(id_seq, action)
#         return [self.abc[i] for i in new_id_seq]

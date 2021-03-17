"""This file defines the environment and the collector used in the environment to collect samples.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from dev_misc import FT, LT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib import pad_to_dense
from dev_misc.devlib.named_tensor import NoName
from dev_misc.utils import handle_sequence_inputs
from sound_law.data.alphabet import PAD_ID, Alphabet, EMP

from .action import SoundChangeAction
# from .agent import AgentInputs, VanillaPolicyGradient
# pylint: disable=no-name-in-module
from .mcts_cpp import PyActionSpaceOpt, PyEnv, PyEnvOpt, PyWordSpaceOpt
# pylint: enable=no-name-in-module
from .trajectory import Trajectory, VocabState


class SoundChangeEnv(PyEnv):

    tnode_cls = VocabState

    add_argument(f'final_reward', default=1.0, dtype=float, msg='Final reward for reaching the end.')
    add_argument(f'step_penalty', default=0.02, dtype=float, msg='Penalty for each step if not the end state.')

    def __init__(self, *args, abc: Alphabet = None, **kwargs):
        self.abc = abc

        # # Set class variable for `SoundChangeAction` here.
        SoundChangeAction.abc = abc

        # Register unconditional actions first.
        units = [u for u in abc if u not in abc.special_units]

        def register_uncondional_action(u1: str, u2: str, cl: bool = False, gb: bool = False):
            id1 = abc[u1]
            id2 = abc[u2]
            if cl:
                self.register_cl_map(id1, id2)
            elif gb:
                if u1.startswith('i'):
                    self.register_gbj_map(id1, id2)
                else:
                    assert u1.startswith('u')
                    self.register_gbw_map(id1, id2)
            else:
                self.register_permissible_change(id1, id2)

        for u1, u2 in abc.edges:
            register_uncondional_action(u1, u2)
        for u in units:
            register_uncondional_action(u, EMP)
        for u1, u2 in abc.cl_map.items():
            register_uncondional_action(u1, u2, cl=True)
        for u1, u2 in abc.gb_map.items():
            register_uncondional_action(u1, u2, gb=True)

        # self.set_vowel_info(abc.vowel_mask, abc.vowel_base, abc.vowel_stress, abc.stressed_vowel, abc.unstressed_vowel)
        # self.set_glide_info(abc['j'], abc['w'])

    def __call__(self, state: VocabState, best_i: int, action: SoundChangeAction) -> Tuple[VocabState, bool, float]:
        return self.step(state, best_i, action)

    def show_path(self, state: VocabState) -> str:
        out = list()
        for action_id, reward in state.get_path():
            action = self.action_space.get_action(action_id)
            out.append(f'{action}, {reward:.3f}')
        return '(' + ', '.join(out) + ')'

    def apply_action(self, state: VocabState, action: SoundChangeAction) -> VocabState:
        return super().apply_action(state,
                                    action.before_id,
                                    action.after_id,
                                    action.rtype,
                                    action.pre_id,
                                    action.d_pre_id,
                                    action.post_id,
                                    action.d_post_id)

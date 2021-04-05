from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (ClassVar, Dict, Iterator, List, NewType, Optional,
                    Sequence, Set, Tuple, Union)

import numpy as np

import sound_law.data.data_loader as dl
import sound_law.rl.action as a
from dev_misc import BT, FT, LT, NDA, g, get_tensor
from dev_misc.devlib import pad_to_dense
from dev_misc.utils import (Singleton, cached_property,
                            is_main_process_and_thread)
from editdistance import eval_batch
from sound_law.data.alphabet import NULL_ID, PAD_ID, Alphabet

from .mcts_cpp import (PyPath, PyST_CLL,  # pylint: disable=no-name-in-module
                       PyST_CLR, PyST_GBJ, PyST_GBW, PyST_NONE, PyST_VS,
                       PyTreeNode, parallel_gather_trajectory)

int2st = {
    PyST_CLL: 'CLL',
    PyST_CLR: 'CLR',
    PyST_NONE: 'basic',
    PyST_VS: 'VS',
    PyST_GBJ: 'GBJ',
    PyST_GBW: 'GBW',
    NULL_ID: None  # This is used in STOP action.
}


def strip_stress(s):
    if '{' in s:
        return s[:-3]
    return s


class VocabState(PyTreeNode):
    """State representing the vocab. Use `VocabStateSpace` to create one instance."""

    abc: ClassVar[Alphabet] = None

    @cached_property
    def tensor(self) -> LT:
        """Convert the state into a long tensor."""
        return get_tensor(self.vocab_array).rename('word', 'pos')

    @property
    def q(self):
        return self.total_values / (1e-8 + self.action_counts)

    @property
    def word_list(self) -> List[str]:
        assert self.abc is not None
        words = list()
        for id_seq in self.vocab:
            words.append(''.join(self.abc[i] for i in id_seq[1:-1]))  # pylint: disable=unsubscriptable-object
        return words

    @property
    def segment_list(self) -> List[List[str]]:
        assert self.abc is not None
        words = list()
        for id_seq in self.vocab:
            words.append([self.abc[i] for i in id_seq])  # pylint: disable=unsubscriptable-object
        return words

    @property
    def alphabet(self) -> List[str]:
        ret = set()
        for segments in self.segment_list:
            ret.update([strip_stress(seg) for seg in segments])
        ret.remove('<SOT>')
        ret.remove('<EOT>')
        return sorted(ret)

    def get_num_occurences(self, unit: str) -> int:
        has_stress = '{' in unit
        ret = 0
        for segments in self.segment_list:
            for seg in segments:
                if has_stress:
                    ret += strip_stress(seg) == unit
                else:
                    ret += seg == unit
        return ret


@dataclass
class TrEdge:
    """This represents one edge in the trajectories."""
    step: int
    s0: VocabState
    a: a.SoundChangeAction
    pa: NDA
    r: float
    qs: NDA
    s1: VocabState
    mcts_pi: NDA
    almt1: Optional[NDA] = None
    almt2: Optional[NDA] = None


class Trajectory:

    def __init__(self, played_path: PyPath, max_end_length: int):
        # NOTE(j_luo) They have different batch size. `id_seqs` has n + 1, `rewards` has n (last state doesn't have any q due to being unexplored), while the remaining tree have 7 * n each.
        if g.repr_mode != 'state':
            self.id_seqs, self.almts1, self.almts2, self.actions, self.rewards, self.permissible_actions, self.mcts_pis, self.qs = parallel_gather_trajectory(
                played_path, g.num_workers, True, max_end_length)
            assert len(self.id_seqs) == len(self.almts1) == len(self.almts2)
        else:
            self.id_seqs, self.actions, self.rewards, self.permissible_actions, self.mcts_pis, self.qs, self.ret = parallel_gather_trajectory(
                played_path, g.num_workers, False, max_end_length)
        # breakpoint()  # BREAKPOINT(j_luo)
        self.done = played_path.get_last_node().done
        self._num_edges = len(self.id_seqs) - 1
        assert len(self.rewards) == len(self.id_seqs) - 1
        assert len(self.actions) == len(self.permissible_actions) == len(
            self.mcts_pis) == len(self.qs) == 7 * self._num_edges
        self.total_reward = self.rewards.sum()

    def __len__(self):
        return self._num_edges

    def __repr__(self):
        return f'Total reward: {self.total_reward:.3f}\tlength {self._num_edges}\t' + ', '.join(f'({edge.a}, {edge.r:.3f})' for edge in self)

    def __iter__(self) -> Iterator[TrEdge]:
        for i in range(self._num_edges):
            s0 = self.id_seqs[i]
            s1 = self.id_seqs[i + 1]
            start = 7 * i
            end = start + 7
            r = self.rewards[i]
            action = a.SoundChangeAction(self.actions[start], self.actions[start + 2],
                                         int2st[self.actions[start + 1]], self.actions[start + 3],
                                         self.actions[start + 4], self.actions[start + 5], self.actions[start + 6])
            qs = self.qs[start:end]
            pa = self.permissible_actions[start: end]
            mcts_pi = self.mcts_pis[start: end]
            almt1 = almt2 = None
            if g.repr_mode != 'state':
                almt1 = self.almts1[i]
                almt2 = self.almts2[i]
            yield TrEdge(i, s0, action, pa, r, qs, s1, mcts_pi, almt1=almt1, almt2=almt2)

from __future__ import annotations

import re
from typing import ClassVar, List, Optional, Union

import pandas as pd

from dev_misc import add_argument, g
from pypheature.process import FeatureProcessor
from sound_law.data.alphabet import Alphabet
from sound_law.main import setup
from sound_law.rl.action import SoundChangeAction, SoundChangeActionSpace
from sound_law.rl.mcts_cpp import PyNull_abc
from sound_law.rl.trajectory import VocabState
from sound_law.train.manager import OnePairManager

_fp = FeatureProcessor()


def named_ph(name: str) -> str:
    ph = ''.join([
        fr'(?P<{name}>',       # named capture group start
        '('
        r'[^+\(\)\[\] ]+',           # acceptable characters (everything except '+', ' ', '(' and ')')
        '|',
        r'\[[\w ,\+\-]+\]',
        ')',
        ')'                   # capture group end
    ])
    return ph


pre_cond_pat = ''.join([
    '(',     # optional start 1
    ' *',    # optional whitespace
    r'\(',    # left parenthesis
    '(',     # optional start 2
    ' *',
    named_ph('d_pre'),      # d_pre
    ' *',    # optional whitespace
    r'\+',    # plus sign
    ' *',    # optional whitespace
    ')?',    # optional end 2
    named_ph('pre'),      # pre
    ' *',
    r'\)',    # right parenthesis
    ' *',    # optional whitespace
    r'\+',    # plus sign
    ' *',    # optional whitespace
    ')?'     # optional end 1
])


post_cond_pat = ''.join([
    '(',     # optional start 1
    ' *',
    r'\+',
    ' *',
    r'\(',    # left parenthesis
    ' *',
    named_ph('post'),
    '(',     # optional start 2
    ' *',
    r'\+',
    ' *',
    named_ph('d_post'),
    ' *',
    ')?',    # optional end 2
    r'\)',    # right parenthesis
    ' *',
    ')?'     # optional end 1
])

pat = re.compile(fr'^{pre_cond_pat}{named_ph("before")}{post_cond_pat} *> *{named_ph("after")} *$')

error_codes = {'BDR', 'SCP', 'NC', 'SS', 'EPTh', 'MS', 'MTTh', 'IRG', 'OPT', 'LD', 'CIS', 'OOS', 'ALPh'}
# A: NW, B: Gothic, C: W, D.1: Ingvaeonic, D.2: AF, E: ON, F: OHG, G: OE
# Gothic: B, ON: A-E, OHG: A-C-F, OE: NW-D.1-D.2-G


class ExpandableAction:
    ...


Action = Union[SoundChangeAction, ExpandableAction]


def get_rules(series, orders, abc: Alphabet) -> List[SoundChangeAction]:
    rules = [None] * len(series)
    for i, (cell, order) in enumerate(zip(series, orders)):
        if not pd.isnull(cell):
            cell_results = list()
            for line in cell.strip().split('\n'):
                # print(line)
                # if set(line) & set("[]"):
                #     continue
                if all(code not in line for code in error_codes):
                    result = pat.match(line)
                    d_pre = result.group('d_pre')
                    pre = result.group('pre')
                    before = result.group('before')
                    post = result.group('post')
                    d_post = result.group('d_post')
                    after = result.group('after')

                    cell_results.append(SoundChangeAction.from_str(before, after, pre, d_pre, post, d_post))
            order = i if pd.isnull(order) else int(order)
            if cell_results:
                rules[order] = cell_results
    ret = list()
    for cell_results in rules:
        if cell_results:
            ret.extend(cell_results)
    return ret


class PlainState:
    """This stores the plain vocabulary state (using str), as opposed to `VocabState` that is used by MCTS."""

    action_space: ClassVar[SoundChangeActionSpace] = None
    end_state: ClassVar[PlainState] = None
    abc: ClassVar[Alphabet] = None

    def __init__(self, segments: List[List[str]]):
        self.segments = segments

    @classmethod
    def from_vocab_state(cls, vocab: VocabState) -> PlainState:
        return cls(vocab.segment_list)

    def apply_action(self, action: SoundChangeAction) -> PlainState:
        cls = type(self)
        assert cls.action_space is not None
        new_segments = list()
        for seg in self.segments:
            new_segments.append(cls.action_space.apply_action(seg, action))
        return cls(new_segments)

    @property
    def dist(self) -> float:
        cls = type(self)
        assert cls.end_state is not None
        assert cls.abc is not None
        dist = 0.0
        for s1, s2 in zip(self.segments, cls.end_state.segments):
            s1 = [cls.abc[u] for u in s1]  # pylint: disable=unsubscriptable-object
            s2 = [cls.abc[u] for u in s2]  # pylint: disable=unsubscriptable-object
            dist += cls.action_space.word_space.get_edit_dist(s1, s2)
        return dist


if __name__ == "__main__":
    # Get alphabet and action space.
    initiator = setup()
    initiator.run()
    manager = OnePairManager()

    # Get the list of rules.
    df = pd.read_csv('data/test_annotations.csv')
    df = df.dropna(subset=['ref no.'])
    got_df_rules = df[df['ref no.'].str.startswith('B')]['v0.4']
    got_rows = df[df['ref no.'].str.startswith('B')]
    gold = get_rules(got_rows['v0.4'], got_rows['order'], manager.tgt_abc)

    # Simulate the actions and get the distance.
    state = PlainState.from_vocab_state(manager.env.start)
    PlainState.action_space = manager.action_space
    PlainState.end_state = PlainState.from_vocab_state(manager.env.end)
    PlainState.abc = manager.tgt_abc
    print(state.dist)
    for action in gold:
        next_state = state.apply_action(action)
        print(action)
        print(next_state.dist)
        state = next_state

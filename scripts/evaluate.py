from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Union

import pandas as pd

from dev_misc import add_argument, g
from pypheature.nphthong import Nphthong
from pypheature.process import FeatureProcessor
from pypheature.segment import Segment
from sound_law.data.alphabet import Alphabet
from sound_law.main import setup
from sound_law.rl.action import SoundChangeAction, SoundChangeActionSpace
from sound_law.rl.mcts_cpp import \
    PyNull_abc  # pylint: disable=no-name-in-module
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


class Boundary:

    def __str__(self):
        return '#'


SegmentLike = Union[Segment, Nphthong, Boundary]


@dataclass
class Expandable:
    raw: Optional[str] = None
    expandable: bool = field(init=False, repr=False)
    fv: List[str] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.raw is not None and '[' in self.raw:
            self.fv = [part.strip() for part in self.raw.strip('[]').split(',')]
            self.expandable = True
        else:
            self.expandable = False

    def exists(self) -> bool:
        return self.raw is not None

    def match(self, segment: SegmentLike) -> bool:
        assert self.exists()
        if isinstance(segment, Nphthong):
            return not self.expandable and self.raw == str(segment)
        if isinstance(segment, Boundary):
            return self.raw == '#'
        if self.expandable:
            return segment.check_features(self.fv)
        return self.raw == str(segment)


@dataclass
class ExpandableAction:
    """This is under-specified and would be specialized into `SoundChangeAction` later and be indexed."""
    before: Expandable
    after: Expandable
    pre: Expandable
    d_pre: Expandable
    post: Expandable
    d_post: Expandable

    def __post_init__(self):
        if self.d_pre is not None and self.pre is None:
            raise ValueError(f"`pre` must be present for `d_pre`.")
        if self.d_post is not None and self.post is None:
            raise ValueError(f"`post` must be present for `d_post`.")

    def specialize(self, state: PlainState) -> List[SoundChangeAction]:
        ret = set()
        for segments in state.segments:
            segments = [Boundary()] + [_fp.process(seg) for seg in segments[1:-1]] + [Boundary()]
            n = len(segments)
            for i, seg in enumerate(segments[1:-1], 1):
                applied = self.before.match(seg)
                if applied and self.pre.exists():
                    applied = i > 0 and self.pre.match(segments[i - 1])
                if applied and self.d_pre.exists():
                    applied = i > 1 and self.d_pre.match(segments[i - 2])
                if applied and self.post.exists():
                    applied = i < n - 1 and self.post.match(segments[i + 1])
                if applied and self.d_post.exists():
                    applied = i < n - 2 and self.d_post.match(segments[i + 2])

                if applied:
                    if self.after.expandable:
                        after = str(_fp.change_features(seg, self.after.fv))
                    else:
                        after = self.after.raw
                    pre = str(segments[i - 1]) if self.pre.exists() else None
                    d_pre = str(segments[i - 2]) if self.d_pre.exists() else None
                    post = str(segments[i + 1]) if self.post.exists() else None
                    d_post = str(segments[i + 2]) if self.d_post.exists() else None
                    ret.add(SoundChangeAction.from_str(str(seg), after, pre, d_pre, post, d_post))
        return list(ret)


Action = Union[SoundChangeAction, ExpandableAction]


def get_action(raw_line: str) -> Action:
    result = pat.match(raw_line)
    d_pre = result.group('d_pre')
    pre = result.group('pre')
    before = result.group('before')
    post = result.group('post')
    d_post = result.group('d_post')
    after = result.group('after')

    if '[' in raw_line:
        return ExpandableAction(Expandable(before), Expandable(after), Expandable(pre), Expandable(d_pre), Expandable(post), Expandable(d_post))
    return SoundChangeAction.from_str(before, after, pre, d_pre, post, d_post)


def get_actions(series, orders) -> List[Action]:
    rules = [None] * len(series)
    for i, (cell, order) in enumerate(zip(series, orders)):
        if not pd.isnull(cell):
            cell_results = list()
            for line in cell.strip().split('\n'):
                if all(code not in line for code in error_codes):
                    cell_results.append(get_action(line))
            order = i if pd.isnull(order) else int(order)
            if cell_results:
                rules[order] = cell_results
    ret = list()
    for cell_results in rules:
        if cell_results:
            ret.extend(cell_results)
    return ret


class PlainState:
    """This stores the plain vocabulary state (using str), as opposed to `VocabState` that is used by MCTS (using ids)."""

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
    add_argument("in_path", dtype=str, msg="Input path to the saved path file.")
    # Get alphabet and action space.
    initiator = setup()
    initiator.run()
    manager = OnePairManager()

    dump = pickle.load(open(g.segments_dump_path, 'rb'))
    _fp.load_repository(dump['proto_ph_map'].keys())

    # Get the list of rules.
    if g.in_path:
        with open(g.in_path, 'r', encoding='utf8') as fin:
            lines = [line.strip() for line in fin.readlines()]
            gold = get_actions(lines, range(len(lines)))

    else:
        df = pd.read_csv('data/test_annotations.csv')
        df = df.dropna(subset=['ref no.'])
        got_df_rules = df[df['ref no.'].str.startswith('B')]['v0.4']
        got_rows = df[df['ref no.'].str.startswith('B')]
        gold = get_actions(got_rows['v0.4'], got_rows['order'])

    # Simulate the actions and get the distance.
    state = PlainState.from_vocab_state(manager.env.start)
    PlainState.action_space = manager.action_space
    PlainState.end_state = PlainState.from_vocab_state(manager.env.end)
    PlainState.abc = manager.tgt_abc
    print(state.dist)
    for action in gold:
        if isinstance(action, SoundChangeAction):
            state = state.apply_action(action)
            print(action)
            print(state.dist)
        else:
            for a in action.specialize(state):
                state = state.apply_action(a)
                print(a)
                print(state.dist)

    # compute the similarity between the candidate ruleset and the gold standard ruleset
    candidate: List[Action] = None # let this be the model's ruleset, which we are comparing to gold 
    # first, what % of the gold ruleset is present in candidate?
    n_shared_actions = 0
    n_similar_actions = 0 # similar actions get half credit. We count separately so these are stored as int
    for action in gold:
        similar_actions = get_similar_actions(action)
        for candidate_act in candidate:
            if candidate_act == action:
                n_shared_actions += 1
            # problem: what if an exact rule and a similar rule are both in candidate? then you could get more than 100% similarity.
            if candidate_act in similar_actions:
                n_similar_actions += 1
    ruleset_containment = (n_shared_actions + (.5 * n_similar_actions)) / len(gold)
    print('candidate ruleset contains ' + str(ruleset_containment) + '\% of the gold rules')
    # is there a way to combine this metric with the one below? i.e., to say that a given rule is only 'partially contained' within candidate if it's present, but in the wrong order relative to other dependent actions [actions it could feed or bleed]?

    # assume that candidate is a ruleset that contains the same rules as the gold ruleset but has them in a different order.
    # first, identify which pairs of actions are in the wrong order in candidate
    # this is a very simple O(n) alg for doing this, but it's not that space efficient since it uses a dictionary. Investigate improvements.
    act_to_index = {act: i for i, act in enumerate(gold)}
    swaps = 0 # number of pairs that are out of order in an impactful way
    # it's easier to think of rules that rule A could feed or bleed than the reverse. We want to identify when A feeds B but B precedes A in order of application; and so the easiest way to do this is to iterate over the rules in reverse.
    # assuming actions are applied in the order 0 to end
    for i, act1 in enumerate(candidate):
        dependent_strings = []

        before_string = act1.before
        after_string = act1.after
        if act1.pre:
            before_string = act1.pre + before_string
            after_string = act1.pre + after_string
            if act1.d_pre:
                before_string = act1.d_pre + before_string
                after_string = act1.d_pre + after_string
        if act1.post:
            before_string = before_string + act1.post
            after_string = after_string + act1.post
            if act1.d_post:
                before_string = before_string + act1.d_post
                after_string = after_string + act1.d_post
        # try all possible cycles of the string, which constitute all possible dependent rule shapes
        for i in range(len(after_string)):
            cycled_str = after_string[i:] + after_string[:i]
            dependent_strings.append(cycled_str)
            dependent_strings.extend(find_simpler_actions(cycled_str)) # this isn't quite right since find_simpler_actions will presumably take an action instead of a string, but work with it for now

        # identify those actions which could be fed/bled by act1 but precede act1 — a relationship that could happen but does not occur
        for act2 in candidate[:i]:
            if act2 in dependent_strings:
                # act2 could possibly be fed/bled by act1, but we must confirm that the order is also incongruous with the gold standard
                if act_to_index[act1] < act_to_index[act2]:
                    swaps += 1
    
    print(str(swaps) + ' pairs of rules in wrong order in candidate')

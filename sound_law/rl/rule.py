# TODO(djwyen) make scripts/evaluate.py import its functions frorm here instead
from __future__ import annotations

import logging
import pickle
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from dev_misc import add_argument, g
from pypheature.nphthong import Nphthong
from pypheature.process import (FeatureProcessor, NoMappingFound,
                                NonUniqueMapping)
from pypheature.segment import Segment
from sound_law.data.alphabet import EOT, SOT, Alphabet
from sound_law.main import setup
from sound_law.rl.action import SoundChangeAction
from sound_law.rl.env import SoundChangeEnv
# from sound_law.rl.mcts_cpp import
#     PyNull_abc  # pylint: disable=no-name-in-module
from sound_law.rl.trajectory import VocabState
from sound_law.train.manager import OnePairManager

_fp = FeatureProcessor()
# NOTE(j_luo) We use `a` to represent the back vowel `ɑ`.
back_a = _fp._base_segments['ɑ']
front_a = deepcopy(back_a)
front_a.base = 'a'
_fp._base_segments['a'] = front_a


syl_info = ''.join([
    '(',
    r'\{',
    r'[\+\-\w, ]+',
    r'\}'
    ')?'
])


def named_ph(name: str) -> str:
    return ''.join([
        fr'(?P<{name}>',       # named capture group start
        '('
        r'[^+\(\)\[\]\{\} ]+',           # acceptable characters (everything except '+', ' ', '(' and ')')
        '|',
        r'\[[\w̃  ,\+\-!]+\]',   # specify natural classes
        ')',
        syl_info,
        ')'                   # capture group end
    ])


pre_cond_pat = ''.join([
    '(',     # optional start 1
    ' *',    # optional whitespace
    '(',     # optional start 2
    ' *',
    named_ph('d_pre'),      # d_pre
    ' +',    # optional whitespace
    ')?',    # optional end 2
    named_ph('pre'),      # pre
    ' *',    # optional whitespace
    ')?'     # optional end 1
])


post_cond_pat = ''.join([
    '(',     # optional start 1
    ' *',
    named_ph('post'),
    '(',     # optional start 2
    ' +',
    named_ph('d_post'),
    ' *',
    ')?',    # optional end 2
    ' *',
    ')?'     # optional end 1
])

rtype_pat = ''.join([
    r'(?P<rtype>',
    'basic',
    '|',
    'VS',
    '|',
    'OGF',
    '|',
    'CLL',
    '|',
    'CLR'
    ')'
])

pat = re.compile(
    # fr'^{pre_cond_pat}{named_ph("before")}{post_cond_pat} *> *{named_ph("after")} *$')
    fr'^{rtype_pat}: *{named_ph("before")} *> *{named_ph("after")} *(/ *{pre_cond_pat} *_ *{post_cond_pat} *)*$')

error_codes = {'UR', 'NP', 'IRG', 'MS', 'MISC', 'DUP'}
# A: NW, B: Gothic, C: W, D.1: Ingvaeonic, D.2: AF, E: ON, F: OHG, G: OE
# Gothic: B, ON: A-E, OHG: A-C-F, OE: A-D.1-D.2-G
ref_no = {
    'got': ['B'],
    'non': ['A', 'E'],
    'ang': ['A', 'C', 'D.1', 'D.2', 'G']
}


stress_pat = re.compile(r'\{[\w, \+\-]+\}')


@dataclass
class HandwrittenSegment:
    """This represents a handwritten segment (including natural class)."""

    def __init__(self, raw_seg: Union[str, None], raw_stress: Union[str, None]):
        raw_seg = None if raw_seg == '' else raw_seg
        raw_stress = None if raw_stress == '' else raw_stress
        if raw_stress:
            segs = [seg.strip() for seg in raw_stress.strip('{}').split(',')]
            segs = [seg for seg in segs if 'heavy' not in seg]
            if segs:
                assert len(segs) == 1
                raw_stress = '{' + segs[0][0] + '}'
            else:
                raw_stress = None

        self._raw_seg = raw_seg
        self._raw_stress = raw_stress
        self.expandable = self._raw_seg is not None and '[' in self._raw_seg
        self.fv = None
        if self.expandable:
            self.fv = [part.strip() for part in raw_seg.strip('[]').split(',')]

        assert self._raw_stress in ['{+}', '{-}', None]

    @classmethod
    def from_str(cls, raw: Union[str, None]) -> HandwrittenSegment:
        if not raw:
            return HandwrittenSegment(None, None)
        raw = raw.strip()

        if '{' in raw:
            raw_stress = stress_pat.search(raw).group()
            raw_seg = raw[:-len(raw_stress)]
        else:
            raw_seg = raw
            raw_stress = None
        return HandwrittenSegment(raw_seg, raw_stress)

    def to_segment(self) -> Union[Nphthong, Segment]:
        assert not self.expandable
        return _fp.process(self._raw_seg)

    def __str__(self):
        if self._raw_seg is None:
            return ''
        else:
            return self._raw_seg + self.stress_str

    def exists(self) -> bool:
        return self._raw_seg is not None

    def has_stress(self) -> bool:
        return self._raw_stress is not None

    def __repr__(self):
        return repr(str(self))

    @property
    def stress_str(self) -> str:
        return '' if self._raw_stress is None else self._raw_stress

    @property
    def segment_str(self) -> str:
        return '' if self._raw_seg is None else self._raw_seg

    def match(self, ph: HS) -> bool:
        if not self.exists():
            return True
        if not ph.exists():
            return False

        if self.has_stress():
            if self.stress_str != ph.stress_str:
                return False

        if self._raw_seg == '.':
            return ph.segment_str not in [EOT, SOT]

        if self._raw_seg == '#':
            return ph.segment_str in [EOT, SOT]

        if self.expandable:
            if ph.segment_str in [EOT, SOT]:
                return False
            seg = _fp.process(ph.segment_str)
            # Special case for `voice`.
            if isinstance(seg, Nphthong):
                if self.fv == ['+voice']:
                    return True

            return not isinstance(seg, Nphthong) and seg.check_features(self.fv)

        assert self._raw_seg != '##'

        return ph.segment_str == self.segment_str


HS = HandwrittenSegment


def get_arg(hs: HS) -> Union[str, None]:
    """Get argument for `SoundChangeAction` initialization."""
    if hs.exists():
        return str(hs)
    return None


@dataclass
class HandwrittenRule:
    """This represents a handwritten rule (from annotations). But it can also represent system-generated rules."""

    before: HS
    after: HS
    rtype: str
    pre: HS
    d_pre: HS
    post: HS
    d_post: HS
    expandable: bool
    ref: Optional[str] = None

    @classmethod
    def from_str(cls, raw: str, ref: Optional[str] = None) -> HandwrittenRule:

        def get_segment(name: str) -> HS:
            return HandwrittenSegment.from_str(result.group(name))

        result = pat.match(raw)
        rtype = result.group('rtype').replace('GB', 'OGF')
        d_pre = get_segment('d_pre')
        pre = get_segment('pre')
        before = get_segment('before')
        post = get_segment('post')
        d_post = get_segment('d_post')
        after = get_segment('after')
        expandable = '[' in raw

        return cls(before, after, rtype, pre, d_pre, post, d_post, expandable, ref=ref)

    def to_action(self) -> SoundChangeAction:
        assert not self.expandable

        def get_arg(seg: HS) -> Union[str, None]:
            ret = str(seg)
            return ret if ret else None

        before = get_arg(self.before)
        after = get_arg(self.after)
        pre = get_arg(self.pre)
        d_pre = get_arg(self.d_pre)
        post = get_arg(self.post)
        d_post = get_arg(self.d_post)

        # There is a minor difference in implementation regarding GB-type rules. Here "GB" is enough, but for c++, two types "GBJ" and "GBW" are used.
        cpp_rtype = self.rtype
        if self.rtype in ['CLL', 'CLR']:
            tgt = self.pre if self.rtype == 'CLL' else self.post
            tgt_seg = tgt.to_segment()
            assert isinstance(tgt_seg, Nphthong) or tgt_seg.is_short()
            after = f'{tgt_seg}ː{tgt.stress_str}'
        elif self.rtype == 'GB':
            before_seg = self.before.to_segment()
            assert isinstance(before_seg, Nphthong)
            assert len(before_seg.vowels) == 2
            first_v = str(before_seg.vowels[0])
            assert first_v in ['i', 'u']
            after = str(before_seg.vowels[1]) + self.before.stress_str
            cpp_rtype = 'GBJ' if first_v == 'i' else 'GBW'

        return SoundChangeAction.from_str(before, after, cpp_rtype, pre, d_pre, post, d_post)

    def specialize(self, state: PlainState) -> List[SoundChangeAction]:
        assert self.expandable
        assert self.rtype != 'GB'

        def safe_get(segments: List[HS], idx: int) -> HS:
            if idx < 0 or idx >= len(segments):
                return HandwrittenSegment.from_str(None)
            return segments[idx]

        def realize(seg: HS, ph: HS) -> HS:
            if seg.exists():
                if seg.has_stress():
                    return ph
                else:
                    return HandwrittenSegment.from_str(ph.segment_str)
            return HandwrittenSegment.from_str(None)

        def is_vowel(ph: HS) -> bool:
            seg = _fp.process(ph.segment_str)
            return isinstance(seg, Nphthong) or seg.is_vowel()

        segs = [self.d_pre, self.pre, self.before, self.post, self.d_post]
        ret = set()
        for segments in state.segments:
            segments = [HandwrittenSegment.from_str(ph) for ph in segments]
            # Deal with vowel sequence.
            if self.rtype == 'VS':
                segments = [segments[0]] + [seg for seg in segments[1:-1] if is_vowel(seg)] + [segments[-1]]

            n = len(segments)
            for i in range(1, n - 1):
                site = [safe_get(segments, i - 2), safe_get(segments, i - 1), safe_get(segments, i),
                        safe_get(segments, i + 1), safe_get(segments, i + 2)]
                if all(hs.match(s) for hs, s in zip(segs, site)):
                    d_pre, pre, before, post, d_post = [realize(hs, s) for hs, s in zip(segs, site)]
                    if self.rtype in ['CLL', 'CLR']:
                        tgt = pre if self.rtype == 'CLL' else post
                        tgt_seg = tgt.to_segment()
                        if not (isinstance(tgt_seg, Nphthong) or tgt_seg.is_short()):
                            continue
                        after = tgt.segment_str + 'ː' + tgt.stress_str
                    elif self.after.expandable:
                        try:
                            after = str(_fp.change_features(before.to_segment(), self.after.fv))
                        except (NonUniqueMapping, NoMappingFound):
                            continue
                    else:
                        after = str(self.after)
                    ret.add(SoundChangeAction.from_str(get_arg(before), after, self.rtype, get_arg(pre), get_arg(d_pre),
                                                       get_arg(post), get_arg(d_post)))
        return list(ret)


def get_actions(raw_rules: List[str],
                orders: Optional[List[str]] = None,
                refs: Optional[List[str]] = None) -> List[HandwrittenRule]:
    if orders is None:
        orders = [None] * len(raw_rules)
    # If `orders` is provided, `refs` must be too.
    else:
        assert refs is not None
        assert len(refs) == len(orders) == len(raw_rules)

    if refs is None:
        refs = list(range(1, len(raw_rules) + 1))

    rules = dict()  # ref-to-rules mapping.
    ordered_refs = list()  # This stores the (chronologically) ordered list of ref numbers.
    for i, (cell, order, ref) in enumerate(zip(raw_rules, orders, refs)):
        assert ref not in rules, 'Duplicate ref number!'
        # If `cell` contains an error code, skip this row.
        if any(code in cell for code in error_codes):
            continue

        # Main body: use `cell_extended` if it's not empty, o/w use `cell`.
        cell_results = list()
        for raw in cell.strip().split('\n'):
            cell_results.append(HandwrittenRule.from_str(raw, ref=ref))

        # When `order` is not null, we need to adjust the rule ordering.
        if not pd.isnull(order):
            assert order in rules
            ordered_refs.insert(ordered_refs.index(order), ref)
        else:
            ordered_refs.append(ref)
        rules[ref] = cell_results
    ret = list()
    for ref in ordered_refs:
        cell_results = rules[ref]
        if cell_results:
            ret.extend(cell_results)
    return ret


class PlainState:
    """This stores the plain vocabulary state (using str), as opposed to `VocabState` that is used by MCTS (using ids)."""

    # action_space: ClassVar[SoundChangeActionSpace] = None
    env: ClassVar[SoundChangeEnv] = None
    end_state: ClassVar[PlainState] = None
    abc: ClassVar[Alphabet] = None

    def __init__(self, node: VocabState):
        self.segments = node.segment_list
        self._node = node

    def apply_action(self, action: SoundChangeAction) -> PlainState:
        cls = type(self)
        assert cls.env is not None
        try:
            new_node = cls.env.apply_action(self._node, action)
            return cls(new_node)

        except RuntimeError:
            logging.warning("No site was targeted.")
            return self

    def dist_from(self, tgt_segments: List[List[str]]):
        '''Returns the distance between the current state and a specified state of segments'''
        cls = type(self)
        assert cls.abc is not None
        assert tgt_segments is not None
        dist = 0.0
        for s1, s2 in zip(self.segments, tgt_segments):
            seq1 = [cls.abc[u] for u in s1]  # pylint: disable=unsubscriptable-object
            seq2 = [cls.abc[u] for u in s2]  # pylint: disable=unsubscriptable-object
            # print(''.join(s1), ''.join(s2), cls.env.get_edit_dist(seq1, seq2))
            dist += cls.env.get_edit_dist(seq1, seq2)
        return dist

    @property
    def dist(self) -> float:
        '''Returns the distance between the current state and the end state'''
        cls = type(self)
        assert cls.end_state is not None
        return self.dist_from(cls.end_state.segments)
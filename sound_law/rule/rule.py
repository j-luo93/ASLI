from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import (ClassVar, Dict, List, Optional, Tuple, TypeVar, Union,
                    overload)

from lingpy.sequence.sound_classes import syllabify
from panphon import FeatureTable
from typeguard import typechecked

from dev_misc.utils import cached_property

ft = FeatureTable()


class RawUnit:

    def __init__(self, raw: str):
        self.raw = raw

    def __repr__(self):
        return self.raw


class Summand:

    @typechecked
    def __add__(self, other: Summand) -> SegSeq:
        segs = list()

        def extend(obj):
            if isinstance(obj, Segment):
                segs.append(obj)
            else:
                segs.extend(obj.segs)

        extend(self)
        extend(other)

        return SegSeq(segs)


class Segment(RawUnit, Summand):

    def __init__(self, raw: str):
        if raw in ['-', '#', '_']:
            raise ValueError(f'Cannot use "{raw}" for normal segments.')
        super().__init__(raw)

    @cached_property
    def features(self) -> Dict[str, int]:
        ret = ft.word_fts(self.raw)
        if len(ret) != 1:
            raise ValueError(f'panphon should return exactly one segment.')
        data = ret[0].data.copy()
        data['*olong'] = 1 if self.raw.endswith('ːː') else -1
        return data

    def get_feature(self, name: str) -> int:
        return self.features[name]


_s2i = {'+': 1, '-': -1}
_i2s = {1: '+', -1: '-'}


@dataclass
class MatchCondition:
    sc: SoundClass
    seg: Segment


T = TypeVar('T')


class Referrable:

    def __init__(self, arg_names: List[str], coref: Optional[int] = None):
        self._arg_names = arg_names
        self.coref = coref

    def ref(self: T, coref: int) -> T:
        cls = type(self)
        kwargs = {
            name: getattr(self, name)
            for name in self._arg_names
        }
        return cls(coref=coref, **kwargs)


class SoundClass(Referrable):

    def __init__(self, specs: Union[Dict[str, int], List[str]], name: Optional[str] = None, coref: Optional[int] = None):
        if isinstance(specs, list):
            specs = {v[1:]: _s2i[v[0]] for v in specs}
        self.specs: Dict[str, int] = specs
        self.name = name
        super().__init__(['specs', 'name'], coref=coref)

    def intersect(self, other: SoundClass, name: Optional[str] = None) -> SoundClass:
        if set(self.specs.keys()) & set(other.specs.keys()):
            raise ValueError(f'Cannot take the intersection with overlapping distinctive feature requirements.')

        specs = self.specs.copy()
        specs.update(other.specs)
        return SoundClass(specs, name=name)

    def __repr__(self):
        if self.name:
            return self.name

        out = list()
        for k, v in self.specs.items():
            sign = _i2s[v]
            out.append(sign + k)
        return '[' + ', '.join(out) + ']'

    @typechecked
    def match(self, seg: Segment) -> MatchCondition:
        return MatchCondition(self, seg)


class WordBoundary(RawUnit):

    def __init__(self):
        super().__init__('#')


class EmptySegment(RawUnit):

    def __init__(self):
        super().__init__('-')


class PlaceholderCondition:
    """This is a placeholder for the before condition."""

    def __init__(self):
        self.followed: List[CanBeBefore] = list()
        self.preceded: List[CanBeBefore] = list()

    def precede(self, segs: List[CanBeBefore]) -> PlaceholderCondition:
        if self.preceded:
            raise RuntimeError(f'This placeholder already has a precede condition.')
        self.preceded = segs
        return self

    def follow(self, segs: List[CanBeBefore]) -> PlaceholderCondition:
        if self.followed:
            raise RuntimeError(f'This placeholder already has a follow condition.')
        self.followed = segs
        return self

    def __repr__(self):
        out = ''
        if self.followed:
            out += ' + '.join(map(repr, self.followed)) + ' + '
        out += '_'
        if self.preceded:
            out += ' + ' + ' + '.join(map(repr, self.preceded))
        return out

    def sub(self, before: List[CanBeBefore]) -> List[CanBeBefore]:
        return self.followed + before + self.preceded


Condition = Union[PlaceholderCondition, MatchCondition]

UNSPECIFIED = object()


class UnderspecifiedSyllable(Referrable):

    @typechecked
    def __init__(self, content: Union[Segment, SoundClass, EmptySegment], stress=UNSPECIFIED, coref: Optional[int] = None):
        self.content = content
        self.stress = stress
        super().__init__(['content', 'stress'], coref=coref)
        if self.stress is not UNSPECIFIED:
            print('stress support incomplete')

    def __repr__(self):
        return f'syl({self.content})'


CanBeBefore = Union[Segment, SoundClass, WordBoundary, UnderspecifiedSyllable]
CanBeAfter = Union[Segment, SoundClass, EmptySegment, UnderspecifiedSyllable]


def _check_validity(before: List[CanBeBefore]):
    # Ensure word boundaries appear at the end (if at all).
    for i, b in enumerate(before):
        if isinstance(b, WordBoundary) and i != len(before) - 1 and i != 0:
            raise ValueError(f'Word Boundaries can only appear at the end or the start.')
    if len(before) == 1 and isinstance(before[0], WordBoundary):
        raise ValueError(f'Before condition cannot be just one word boundary.')


class SoundChangeRule:

    @typechecked
    def __init__(self, before: List[CanBeBefore], after: List[CanBeAfter], unless: Optional[List[Condition]] = None):
        # unless conditions do NOT take part in `after` -- it only specifies conditions that should not be matched amended for `before`.
        _check_validity(before)
        self.before = before
        self.after = after

        # Check unless conditions are well-formed.
        # IDEA(j_luo) need to go through other tests added below here.
        unless = unless or list()
        for unl in unless:
            if isinstance(unl, PlaceholderCondition):
                _check_validity(unl.sub(before))

        # Check syllables are matched.
        syls_b = [syl for syl in before if isinstance(syl, UnderspecifiedSyllable)]
        syls_a = [syl for syl in after if isinstance(syl, UnderspecifiedSyllable)]

        # Ensure empty segments do NOT appear in `before`.
        for syl in before:
            if isinstance(syl, UnderspecifiedSyllable) and isinstance(syl.content, EmptySegment):
                raise ValueError(f'Empty segments should NOT appear in `before`.')

        def check_syls_are_found(syls_1, syls_2):
            for syl_1 in syls_1:
                if not any((syl_1 is syl_2) or (syl_1.coref is not None and syl_1.coref == syl_2.coref) for syl_2 in syls_2):
                    raise ValueError(f'Unmatched syllables.')

        check_syls_are_found(syls_b, syls_a)
        check_syls_are_found(syls_a, syls_b)

        # Ensure ref numbers are used only once in `before`.
        ref_nums = set()
        for c in before:
            if hasattr(c, 'coref') and c.coref is not None and c.coref in ref_nums:
                raise ValueError(f'Duplicate ref numbers in `before`.')

        self.unless = unless

        self.refs: Dict[int, Referrable] = dict()
        for c in before:
            if isinstance(c, Referrable) and c.coref is not None:
                if c.coref in self.refs:
                    raise ValueError(f'Duplicate corefs for before condition.')
                self.refs[c.coref] = c

    def __repr__(self):
        out = ' + '.join(map(repr, self.before)) + ' > ' + ' + '.join(map(repr, self.after))
        if self.unless:
            out += ', unless ' + ' or '.join(map(repr, self.unless))
        return out


class SegSeq(Summand):

    def __init__(self, segs: List[Segment]):
        self.segs = segs

    def __repr__(self):
        return ' '.join(map(repr, self.segs))

    def __len__(self):
        return len(self.segs)

    @cached_property
    def syllables(self) -> List[SegSeq]:
        syls = syllabify([seg.raw for seg in self.segs], output='nested')
        ret = list()
        for i, syl in enumerate(syls):
            ret.append(SegSeq([Segment(raw) for raw in syl]))
        return ret

    @classmethod
    def from_tokenized(cls, tokens: str) -> SegSeq:
        return cls([Segment(raw) for raw in tokens.split()])

    def match_before(self, pos: int, before: List[CanBeBefore]) -> Tuple[bool, dict, int]:
        values = dict()
        matched_len = 0

        def get_syllable(pos: int, get_next: bool = False) -> Tuple[Union[None, Segseq], int]:
            if pos < -2 or pos >= len(self):
                raise ValueError(f'Position value invalid.')
            if pos == -1:
                return self.syllables[0], len(self.syllables[0])
            # This could be sped up.
            cum_len = 0
            for i, syl in enumerate(self.syllables):
                cum_len += len(syl)
                if cum_len > pos:
                    break
            if get_next:
                if i == len(self.syllables) - 1:
                    return None, None
                syl = self.syllables[i + 1]
                return syl, cum_len + len(syl)
            else:
                return syl, cum_len

        def match_sc(seg: Segment, sc: SoundClass) -> bool:
            for k, v in sc.specs.items():
                if seg.get_feature(k) != v:
                    return False
            return True

        # What about a + C + C > b, where C appears twice but differently?

        def value_exist(seg, c):
            return id(c) in values

        for ri, c in enumerate(before):
            if isinstance(c, WordBoundary):
                if ri == len(before) - 1 and pos != len(self.segs):
                    return False, None, None
                if ri == 0:
                    if pos != 0:
                        return False, None, None
                    continue  # Skip incrementing pos.
            else:
                matched_len += 1
                if pos >= len(self.segs):
                    return False, None, None
                seg = self.segs[pos]
                if isinstance(c, Segment):
                    if seg.raw != c.raw:
                        return False, None, None
                elif isinstance(c, SoundClass):
                    if not match_sc(seg, c):
                        return False, None, None
                    if value_exist(seg, c):
                        if values[id(c)].raw != seg.raw:
                            return False, None, None
                    else:
                        values[id(c)] = seg
                else:
                    # Case: UndersUnderspecifiedSyllable
                    if value_exist(seg, c):
                        if values[id(c).raw] != seg.raw:
                            return False, None, None
                    else:
                        # HACK
                        if ri == 0:
                            syl, next_pos = get_syllable(pos, get_next=False)
                        else:
                            syl, next_pos = get_syllable(pos - 1, get_next=True)
                        if syl is None:
                            return False, None, None
                        to_save = [syl]
                        cc = c.content
                        for si, s in enumerate(syl.segs):
                            if isinstance(cc, Segment):
                                if cc.raw == s.raw:
                                    to_save.append(si)
                            else:
                                if match_sc(s, cc):
                                    to_save.append(si)
                        if len(to_save) == 1:
                            return False, None, None

                        values[id(c)] = to_save
                        pos = next_pos
                        matched_len += len(syl) - 1
                        continue

            pos += 1
        return True, values, matched_len

    def apply_rule(self, rule: SoundChangeRule) -> SegSeq:

        def match_right(values) -> List[Segment]:

            def set_features(seg, c):
                # Set features.
                try:
                    ol, l = c.specs['*olong'], c.specs['long']
                    raw = seg.raw.strip('ːː')
                    if (ol, l) == (1, 1):
                        raw += 'ːː'
                    elif (ol, l) == (-1, 1):
                        raw += 'ː'
                except KeyError:
                    print('set_features support incomplete.')
                    raw = seg.raw
                return Segment(raw)

            segs = list()
            for c in rule.after:
                if isinstance(c, EmptySegment):
                    pass
                elif isinstance(c, Segment):
                    segs.append(c)
                elif isinstance(c, SoundClass):
                    if c.coref is not None:
                        orig_seg = values[id(rule.refs[c.coref])]
                        segs.append(set_features(orig_seg, c))
                    else:
                        segs.append(values[id(c)])
                else:
                    if c.coref is not None:
                        loaded = values[id(rule.refs[c.coref])]
                    else:
                        loaded = values[id(c)]
                    syl_segs = loaded[0].segs
                    to_remove = list()
                    for si in loaded[1:]:
                        if isinstance(c.content, Segment):
                            syl_segs[si] = c.content
                        elif isinstance(c.content, SoundClass):
                            syl_segs[si] = set_features(syl_segs[si], c.content)
                        else:
                            to_remove.append(si)
                    for si in reversed(to_remove):
                        syl_segs.pop(si)
                    segs.extend(syl_segs)

            return segs

        def get_real_length(before: List[CanBeBefore]) -> int:
            if not before:
                return 0
            initial = isinstance(before[0], WordBoundary)
            final = isinstance(before[-1], WordBoundary)
            return len(before) - initial - final + (initial and final and before[0] is before[-1])

        def match_amendment(pos: int, amendment: List[CanBeBefore]) -> bool:
            if amendment:
                matched, _, _ = self.match_before(pos, amendment)
                return matched
            return True

        def match_unless(pos: int, values: dict) -> bool:
            for unl in rule.unless:
                if isinstance(unl, PlaceholderCondition):
                    m_f = match_amendment(pos - get_real_length(unl.followed), unl.followed)
                    m_p = match_amendment(pos + get_real_length(rule.before), unl.preceded)
                    if m_f and m_p:
                        return True
                else:
                    if values[id(unl.sc)].raw == unl.seg.raw:
                        return True
            return False

        i = 0
        out_segs = list()
        while i < len(self.segs):
            seg = self.segs[i]
            # Check if condition is met.
            matched, values, matched_len = self.match_before(i, rule.before)
            if not matched or match_unless(i, values):
                out_segs.append(seg)
                i += 1
            else:
                out_segs.extend(match_right(values))
                i += matched_len

        return SegSeq(out_segs)

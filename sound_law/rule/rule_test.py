from functools import partial

from dev_misc import TestCase, test_with_arguments

from .rule import (PlaceholderCondition, Segment, SoundChangeRule, SoundClass, WordBoundary, EmptySegment, SegSeq,
                   UnderspecifiedSyllable)
from .common import IJ, NHV, LV, OLV

syl = UnderspecifiedSyllable
usyl = partial(UnderspecifiedSyllable, stress=False)
rule = SoundChangeRule

pc = PlaceholderCondition

_b = WordBoundary()
empty = EmptySegment()

s_IJ = syl(IJ)
s_NHV = syl(NHV)
s_empty = syl(empty)


class TestRules(TestCase):

    def _check_ans(self, r: rule, inp: str, ans: str):
        out = SegSeq.from_tokenized(inp).apply_rule(r)
        self.assertEqual(str(out), ans)

    def test_i_mutation(self):
        i_mutation_1 = rule([Segment('ɑ'), s_IJ], [Segment('æ'), s_IJ])
        i_mutation_2 = rule([Segment('o'), s_IJ], [Segment('ø'), s_IJ])
        i_mutation_3 = rule([Segment('u'), s_IJ], [Segment('y'), s_IJ])
        self._check_ans(i_mutation_1, 'ɑ b i n', 'æ b i n')
        self._check_ans(i_mutation_1, 'ɑ j ɑ n', 'æ j ɑ n')
        self._check_ans(i_mutation_2, 'o b i n', 'ø b i n')
        self._check_ans(i_mutation_2, 'o j ɑ n', 'ø j ɑ n')
        self._check_ans(i_mutation_3, 'u b i n', 'y b i n')
        self._check_ans(i_mutation_3, 'u j ɑ n', 'y j ɑ n')

    def test_a_mutation(self):
        a_mutation = rule([Segment('o'), s_NHV], [Segment('u'), s_NHV])
        self._check_ans(a_mutation, 'o b i n', 'o b i n')
        self._check_ans(a_mutation, 'o b ɑ n', 'u b ɑ n')

    def test_final_syl_shorten(self):
        final_syl_shorten = rule([syl(Segment('ɔː')).ref(1), _b], [syl(Segment('u')).ref(1)])
        self._check_ans(final_syl_shorten, 'u b ɔː', 'u b u')
        self._check_ans(final_syl_shorten, 'u d ɔː l', 'u d u l')
        self._check_ans(final_syl_shorten, 'u d ɔː l i n', 'u d ɔː l i n')

    def test_overlong_loss(self):
        overlong_loss = rule([OLV.ref(1)], [LV.ref(1)])
        self._check_ans(overlong_loss, 'ɑːː b', 'ɑː b')
        self._check_ans(overlong_loss, 'b ɑːː', 'b ɑː')
        self._check_ans(overlong_loss, 'b ɔː n', 'b ɔː n')

    def test_simple(self):
        ɛː2ɑː = rule([Segment('ɛː')], [Segment('ɑː')])
        rhotacization = rule([Segment('z')], [Segment('r')])

        self._check_ans(ɛː2ɑː, 'b ɛː n', 'b ɑː n')
        self._check_ans(ɛː2ɑː, 'b ɛ n', 'b ɛ n')
        self._check_ans(rhotacization, 'z ɑ n', 'r ɑ n')
        self._check_ans(rhotacization, 'h ɑu z', 'h ɑu r')

    # ɑi2eː = rule([usyl(ɑi).ref(1)], [usyl(eː).ref(1)])
    # ɑu2oː = rule([usyl(ɑu).ref(1)], [usyl(oː).ref(1)])

    def test_final_z_loss(self):
        final_z_loss = rule([Segment('z'), _b], [empty])
        self._check_ans(final_z_loss, 'z ɑ n', 'z ɑ n')
        self._check_ans(final_z_loss, 'ɑ j ɑ z o n', 'ɑ j ɑ z o n')
        self._check_ans(final_z_loss, 'b i n ɑ z', 'b i n ɑ')

    # # This cannot handle diphthongs.
    # gemination = rule([SV, C, j], [SV, C, C], unless=[C.match(r)])

    # # Nasal is not set.
    # nasal_spirant = rule([V.ref(1), N, F], [NLV.ref(1), F])

    # # ɑ n n not passed.
    # brightening_1 = rule([ɑ], [æ], unless=[pc().precede([C, C]), pc().precede([syl(BV)])])
    # brightening_2 = rule([ɑː], [æː], unless=[pc().precede([w])])

    def test_final_syl_loss(self):
        final_syl_loss_1 = rule([syl(Segment('æ')).ref(1), _b], [s_empty.ref(1)])
        final_syl_loss_2 = rule([syl(Segment('ɑ')).ref(1), _b], [s_empty.ref(1)])
        final_syl_loss_3 = rule([syl(Segment('ɑ̃')).ref(1), _b], [s_empty.ref(1)])
        self._check_ans(final_syl_loss_1, 'b æ n ɑ', 'b æ n ɑ')
        self._check_ans(final_syl_loss_1, 'o r o n ɡ æ', 'o r o n ɡ')
        self._check_ans(final_syl_loss_2, 'b ɑ n ɑ', 'b ɑ n')
        self._check_ans(final_syl_loss_2, 'o r ɑ n ɡ ɑ', 'o r ɑ n ɡ')
        self._check_ans(final_syl_loss_3, 'o r ɑ̃ ɡ ɑ', 'o r ɑ̃ ɡ ɑ')
        self._check_ans(final_syl_loss_3, 'o r ɑ ɡ ɑ̃', 'o r ɑ ɡ')

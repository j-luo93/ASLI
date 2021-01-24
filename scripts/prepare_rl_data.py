import pickle
import re
import sys
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import List

import pandas as pd
from cltk.phonology.old_english.orthophonology import \
    OldEnglishOrthophonology as oe
from cltk.phonology.old_norse.orthophonology import on
from lingpy.sequence.sound_classes import ipa2tokens

from pypheature.nphthong import Nphthong
from pypheature.process import FeatureProcessor
from xib.aligned_corpus.transcriber import RuleBasedTranscriber


@lru_cache(maxsize=None)
def PGmc_ipa_trans(word: str) -> str:  # only for latin-transliterated Gothic and Greek without diacritics
    # NOTE(j_luo) Based on Frederik's code, with minor modifications.
    word = word.lower()
    word = word.replace('₂', '')
    # vowels
    word = re.sub(r"ē", "eː", word)
    word = re.sub(r"ō", "oː", word)
    word = re.sub(r"ā", "aː", word)
    word = re.sub(r"ī", "iː", word)
    word = re.sub(r"ū", "uː", word)

    word = re.sub(r"ô", "oːː", word)
    word = re.sub(r"ê", "eːː", word)

    word = re.sub(r'ǭ', 'õː', word)
    word = re.sub(r'ą', 'ã', word)
    word = re.sub(r'į̄', 'ĩː', word)

    # consonants
    word = re.sub(r"h", "x", word)
    word = re.sub(r"f", "f", word)
    word = re.sub(r"xw", "xʷ", word)
    word = re.sub(r"kw", "kʷ", word)
    word = re.sub(r"þ", "θ", word)

    # alternations
    word = re.sub(r"d", "ð", word)
    word = re.sub(r"nð", "nd", word)
    word = re.sub(r"lð", "ld", word)
    word = re.sub(r"zð", "zd", word)
    word = re.sub(r"^ð", "d", word)

    word = re.sub(r"b", "β", word)
    word = re.sub(r"^β", "b", word)

    word = re.sub(r"g", "ɡ", word)
    word = re.sub(r"ɡw", "ɡʷ", word)

    word = re.sub(r"nk", "ŋk", word)
    word = re.sub(r"ng", "ŋɡ", word)
    word = re.sub(r"ng", "ŋɡ", word)

    return word


got_map = {
    '𐌰': 'a',
    '𐌱': 'b',
    '𐌲': 'g',
    '𐌳': 'd',
    '𐌴': 'e',
    '𐌵': 'q',
    '𐌶': 'z',
    '𐌷': 'h',
    '𐌸': 'þ',
    '𐌹': 'i',
    '𐌺': 'k',
    '𐌻': 'l',
    '𐌼': 'm',
    '𐌽': 'n',
    '𐌾': 'j',
    '𐌿': 'u',
    '𐍀': 'p',
    '𐍂': 'r',
    '𐍃': 's',
    '𐍄': 't',
    '𐍅': 'w',
    '𐍆': 'f',
    '𐍇': 'x',
    '𐍈': 'ƕ',
    '𐍉': 'o',
}


def got_transliterate(s: str) -> str:
    ret = ''
    for c in s:
        ret += got_map[c]
    return ret


def i2t(s):
    tokens = ipa2tokens(s, merge_vowels=True, merge_geminates=True)
    ret = list()
    for token in tokens:
        l = len(token)
        # NOTE(j_luo) Merge geminates into one segment.
        if l % 2 == 0 and token[:l // 2] == token[l // 2:]:
            ret.append(token[:l // 2] + 'ː')
        else:
            ret.append(token)
    return ret


def show_all_segs(series):
    segs = set()
    for tokens in series:
        segs.update(tokens)
    print(' '.join(sorted(segs)))


to_break_got = {
    't͡s': ['t', 's'],
    'ɛːa': ['ɛː', 'a']
}
to_break_pgm = {
    'eːa': ['eː', 'a']
}
to_break = {
    'got': to_break_got,
    'pgm': to_break_pgm,
    'ang': dict(),
    'non': dict()
}


def break_false_complex(s: List[str], lang: str = None) -> List[str]:
    assert lang is not None
    ret = list()
    for seg in s:
        if seg in to_break[lang]:
            ret.extend(to_break[lang][seg])
        else:
            ret.append(seg)
    return ret


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('lang', help="Daughter language.")
    args = parser.parse_args()

    # Get Proto-Germanic cognate data extracted from Wiktionary.
    gem_pro = pd.read_csv('data/gem-pro.tsv', sep='\t')
    # Get Swadesh list.
    swa = pd.read_csv('data/swadesh_gem_pro.tsv', sep='\t', header=None)
    # Remove any duplicates or words that do not have a unique reflex.
    to_keep = set()
    for tokens in swa[2]:
        for token in tokens.split():
            to_keep.add(token.strip('*'))
    kept = gem_pro[gem_pro['gem-pro'].isin(to_keep)].reset_index(drop=True)
    desc = kept[kept['desc_lang'] == args.lang].reset_index(drop=True)
    dups = {k for k, v in desc['gem-pro'].value_counts().to_dict().items() if v > 1}
    desc = desc[~desc['gem-pro'].isin(dups)].reset_index(drop=True)

    # IPA transcription.
    if args.lang == "got":
        ipa_col = 'got_ipa'
        form_col = 'latin'
        got_tr = RuleBasedTranscriber('got')
        desc[form_col] = desc['desc_form'].apply(got_transliterate)
        desc[ipa_col] = desc[form_col].apply(got_tr.transcribe).apply(lambda s: i2t(list(s)[0]))
    elif args.lang == 'ang':
        ipa_col = 'ang_ipa'
        form_col = 'desc_form'
        # NOTE(j_luo) Use the simple `a` phoneme to conform to other transcribers.
        desc[ipa_col] = desc[form_col].apply(lambda s: oe(
            s.strip('-')).replace('g', 'ɡ')).apply(i2t).apply(lambda s: [ss.replace('ɑ', 'a') for ss in s])
    elif args.lang == 'non':
        ipa_col = 'non_ipa'
        form_col = 'desc_form'
        # NOTE(j_luo) Use the simple `a` phoneme to conform to other transcribers.
        # desc[ipa_col] = desc[form_col].apply(on.transcribe).str.replace(
        #     'g', 'ɡ').str.replace('ɸ', 'f').str.replace('h', 'x').apply(i2t).str.replace('')
        to_rectify = [('g', 'ɡ'), ('gʷ', 'ɡʷ'), ('h', 'x'), ('hʷ', 'xʷ'), ('ɛ', 'e'), ('ɣ', 'ɡ')]

        def replace(s: str) -> str:
            for x, y in to_rectify:
                s = s.replace(x, y)
            return s
        desc[ipa_col] = desc[form_col].apply(on.transcribe).apply(i2t).apply(lambda lst: [replace(x) for x in lst])

    else:
        raise ValueError(f'Unrecognized language "{args.lang}".')

    # Get rid of false complex segments.
    show_all_segs(desc[ipa_col])
    desc[ipa_col] = desc[ipa_col].apply(break_false_complex, lang=args.lang)
    show_all_segs(desc[ipa_col])

    desc['pgm_ipa'] = desc['gem-pro'].apply(PGmc_ipa_trans).apply(i2t)
    show_all_segs(desc['pgm_ipa'])
    desc['pgm_ipa'] = desc['pgm_ipa'].apply(break_false_complex, lang='pgm')
    show_all_segs(desc['pgm_ipa'])

    src_df = pd.DataFrame()
    src_df['transcription'] = desc['gem-pro']
    src_df['ipa'] = desc['pgm_ipa'].apply(''.join)
    src_df['tokens'] = desc['pgm_ipa'].apply(' '.join)
    src_df['split'] = 'train'
    src_df.to_csv('test_src.tsv', sep='\t', index=None)
    tgt_df = pd.DataFrame()
    tgt_df['transcription'] = desc[form_col]
    tgt_df['ipa'] = desc[ipa_col].apply(''.join)
    tgt_df['tokens'] = desc[ipa_col].apply(' '.join)
    tgt_df['split'] = 'train'
    tgt_df.to_csv('test_tgt.tsv', sep='\t', index=None)

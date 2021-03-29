import pickle
import re
import sys
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import List, Tuple

import pandas as pd
import streamlit as st
from cltk.phonology.old_english.orthophonology import \
    OldEnglishOrthophonology as oe
from cltk.phonology.old_norse.orthophonology import on
from lingpy.sequence.sound_classes import ipa2tokens

from pypheature.nphthong import Nphthong
from pypheature.process import FeatureProcessor
from sound_law.utils import run_section, run_with_argument
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
    out = ', '.join(sorted(segs))
    if st._is_running_with_streamlit:
        st.write(out)
        st.write(f'Number of sounds: {len(segs)}')
    else:
        print(out)
        print(f'Number of sounds: {len(segs)}')


to_break_got = {
    't͡s': ['t', 's'],
    'ɛːa': ['ɛː', 'a']
}
to_break_pgm = {
    'eːa': ['eː', 'a'],
    'oːa': ['oː', 'a']
}
to_break = {
    'got': to_break_got,
    'pgm': to_break_pgm,
    'ang': dict(),
    'non': dict()
}

got2ipa_map = {
    'g': 'ɡ',
    "ah": "aːh",
    "aih": "ɛh",
    "air": "ɛr",
    "ai": "ɛː",
    "auh": "ɔh",
    "aur": "ɔr",
    "au": "ɔː",
    "ei": "iː",
    "e": "eː",
    "o": "oː",
    "ur": "uːr",
    "uh": "uːh",
    "ab": "aβ",
    "ɛb": "ɛβ",
    "ɔb": "ɔβ",
    "ib": "iβ",
    "eb": "eβ",
    "ob": "oβ",
    "ub": "uβ",
    "bd": "βd",
    "bn": "βn",
    "bm": "βm",
    "bɡ": "βɡ",
    "bl": "βl",
    "bj": "βj",
    "br": "βr",
    "bw": "βw",
    "bz": "βz",
    " β": " b",
    "ad": "að",
    "ɛd": "ɛð",
    "ɔd": "ɔð",
    "id": "ið",
    "ed": "eð",
    "od": "oð",
    "ud": "uð",
    "db": "ðb",
    "dβ": "ðβ",
    "dn": "ðn",
    "dm": "ðm",
    "dɡ": "ðɡ",
    "dl": "ðl",
    "dj": "ðj",
    "dr": "ðr",
    "dw": "ðw",
    "dz": "ðz",
    " ð": " d",
    # "f": "ɸ",
    "f": "f",
    "ɡw": "ɡʷ",
    "hw": "hʷ",
    "aɡ": "aɣ",
    "ɛɡ": "ɛɣ",
    "ɔɡ": "ɔɣ",
    "iɡ": "iɣ",
    "eɡ": "eɣ",
    "oɡ": "oɣ",
    "uɡ": "uɣ",
    "ɡb": "ɣb",
    "ɡβ": "ɣβ",
    "ɡn": "ɣn",
    "ɡm": "ɣm",
    "ɡɡ": "ŋɡ",
    "ɡl": "ɣl",
    "ɡj": "ɣj",
    "ɡr": "ɣr",
    "ɡw": "ɣw",
    "ɡz": "ɣz",
    "ɡp": "xp",
    "ɡt": "xt",
    "ɡk": "ŋk",
    "ɡɸ": "xɸ",
    "ɡh": "xh",
    "ɡs": "xs",
    "ɡþ": "xþ",
    "ɡq": "xq",
    " ɣ": " ɡ",
    " x": " ɡ",
    "qw": "kʷ",
    "þ": "θ",
    'ƕ': 'hʷ',
    'q': 'kʷ'
}


def replace(s: str, repl_map: List[Tuple[str, str]]) -> str:
    for x, y in repl_map:
        s = s.replace(x, y)
    return s


def got_transcribe(s: str) -> str:
    return replace(s, got2ipa_map.items())


def break_false_complex(s: List[str], lang: str = None) -> List[str]:
    assert lang is not None
    ret = list()
    for seg in s:
        if seg in to_break[lang]:
            ret.extend(to_break[lang][seg])
        else:
            ret.append(seg)
    return ret


PDF = pd.DataFrame


@run_section('Loading data...', 'Loading done.')
def load_data(gem_pro_path: str, swadesh_gem_pro_path: str) -> Tuple[PDF, PDF]:
    # Get cognate data.
    gem_pro = pd.read_csv(gem_pro_path, sep='\t')
    # Get Swadesh list.
    swa = pd.read_csv(swadesh_gem_pro_path, sep='\t', header=None)
    return gem_pro, swa


@run_section('Removing any duplicates or words that do not have a unique reflex...',
             'Removal done.')
def remove_duplicate(gem_pro: PDF, swa: PDF) -> PDF:
    to_keep = set()
    for tokens in swa[2]:
        for token in tokens.split():
            to_keep.add(token.strip('*'))
    kept = gem_pro[gem_pro['gem-pro'].isin(to_keep)].reset_index(drop=True)
    desc = kept[kept['desc_lang'] == lang].reset_index(drop=True)
    dups = {k for k, v in desc['gem-pro'].value_counts().to_dict().items() if v > 1}
    desc = desc[~desc['gem-pro'].isin(dups)].reset_index(drop=True)
    return desc


if __name__ == "__main__":
    parser = ArgumentParser()
    st.title('Prepare RL dataset.')
    st.header('Specify the arguments first:')
    lang = run_with_argument('lang',
                             parser=parser,
                             default='',
                             msg='Daughter language.')
    out_dir = run_with_argument('out_dir',
                                parser=parser,
                                default='data/wikt',
                                msg='Output directory')
    gem_pro_path = run_with_argument('gem_pro_path',
                                     parser=parser,
                                     default='data/gem-pro.tsv',
                                     msg='Path to the Proto-Germanic cognate data extracted from Wiktionary.')
    swadesh_gem_pro_path = run_with_argument('swadesh_gem_pro_path',
                                             parser=parser,
                                             default='data/swadesh_gem_pro.tsv',
                                             msg='Path to the Proto-Germanic Swadesh list.')
    gem_pro, swa = load_data(gem_pro_path, swadesh_gem_pro_path)
    desc = remove_duplicate(gem_pro, swa)
    st.write(f'{len(desc)} entries in total')

    if lang == "got":
        ipa_col = 'got_ipa'
        form_col = 'latin'
        desc = desc.assign(**{form_col: desc['desc_form'].apply(got_transliterate)})
        desc = desc.assign(**{ipa_col: desc[form_col].apply(got_transcribe).apply(i2t)})
    elif lang == 'ang':
        ipa_col = 'ang_ipa'
        form_col = 'desc_form'
        # NOTE(j_luo) Use the simple `a` phoneme to conform to other transcribers.
        to_rectify = [('ɑ', 'a'), ('g', 'ɡ'), ('h', 'x'), ('hʷ', 'xʷ'), ('ç', 'x')]

        desc[ipa_col] = desc[form_col].apply(lambda s: oe(
            s.strip('-').replace('ċ', 'c').replace('ġ', 'g'))).apply(i2t).apply(lambda lst: [replace(x, to_rectify) for x in lst])
    elif lang == 'non':
        ipa_col = 'non_ipa'
        form_col = 'desc_form'
        to_rectify = [('g', 'ɡ'), ('gʷ', 'ɡʷ'), ('h', 'x'), ('hʷ', 'xʷ'), ('ɛ', 'e'), ('ɣ', 'ɡ'), ('ɔ', 'o')]
        desc[ipa_col] = desc[form_col].apply(on.transcribe).apply(
            i2t).apply(lambda lst: [replace(x, to_rectify) for x in lst])

    else:
        raise ValueError(f'Unrecognized language "{lang}".')
    st.write(desc)

    # Get rid of false complex segments.
    show_all_segs(desc[ipa_col])
    desc[ipa_col] = desc[ipa_col].apply(break_false_complex, lang=lang)
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
    src_out_path = f'{out_dir}/pgmc-{lang}/pgmc.tsv'
    src_df.to_csv(src_out_path, sep='\t', index=None)
    st.write(f'Source written to {src_out_path}.')

    tgt_df = pd.DataFrame()
    tgt_df['transcription'] = desc[form_col]
    tgt_df['ipa'] = desc[ipa_col].apply(''.join)
    tgt_df['tokens'] = desc[ipa_col].apply(' '.join)
    tgt_df['split'] = 'train'
    tgt_out_path = f'{out_dir}/pgmc-{lang}/{lang}.tsv'
    tgt_df.to_csv(tgt_out_path, sep='\t', index=None)
    st.write(f'Target written to {tgt_out_path}.')

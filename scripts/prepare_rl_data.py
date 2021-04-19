import pickle
import re
import sys
import unicodedata
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from cltk.data.fetch import FetchCorpus
from cltk.phonology.ang.phonology import Transcriber as AngTranscriber
from cltk.phonology.lat.transcription import Transcriber as LatTranscriber
# from cltk.phonology.old_english.orthophonology import \
# OldEnglishOrthophonology as oe
# from cltk.phonology.old_norse.orthophonology import on
from cltk.phonology.non.phonology import OldNorseTranscription
from epitran import Epitran
from ipapy.ipastring import IPAString
from lingpy.sequence.sound_classes import ipa2tokens
from pandas.core.algorithms import isin
from pypheature.nphthong import Nphthong
from pypheature.process import FeatureProcessor
from pypheature.segment import Segment
from sound_law.utils import run_section, run_with_argument
from tensorflow.python.util.nest import flatten_with_joined_string_paths

# from xib.aligned_corpus.transcriber import RuleBasedTranscriber

_processor = FeatureProcessor()


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


def i2t(s, merge_vowels: bool = True):
    tokens = ipa2tokens(s, merge_vowels=merge_vowels, merge_geminates=True)
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
to_break_ru = {
    'ea': ['e', 'a'],
    'oa': ['o', 'a'],
    # According to https://en.wikipedia.org/wiki/Russian_phonology, /n/ and /nʲ/ are the only consonants that can be geminated within morpheme boundaries.
    'ʂʲː': ['ʂʲ', 'ʂʲ']
}
to_break_uk = {
    'iɑ': ['i', 'ɑ'],
    'ɪɑ': ['ɪ', 'ɑ'],
    'ɔɑ': ['ɔ', 'ɑ']
}
to_break = {
    'got': to_break_got,
    'pgm': to_break_pgm,
    'ang': dict(),
    'non': dict(),
    'it': dict(),
    'la': dict(),
    'es': dict(),
    'fr': dict(),
    'ru': to_break_ru,
    'sla-pro': dict(),
    # According to https://arxiv.org/pdf/0802.4198.pdf, there are no phonemic diphthongs.
    'uk': to_break_uk,
    # The Wikipedia page for this did not mention geminates.
    'pl': dict()
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
    return replace(s, list(got2ipa_map.items()))


proto_slavic_map = {
    'ь': 'i',
    'ъ': 'u',
    'e': 'e',
    'o': 'o',
    'i': 'iː',
    'y': 'yː',
    'u': 'uː',
    'ě': 'eː',
    'a': 'aː',
    'ę': 'ẽː',
    'ǫ': 'õː',
    'm': 'm',
    'n': 'n',
    'ň': 'nʲ',
    'p': 'p',
    'b': 'b',
    't': 't',
    'dz': 'd͡z',
    'd': 'd',
    'ť': 'tʲː',
    'ď': 'dʲː',
    'k': 'k',
    'g': 'ɡ',
    'c': 't͡s',
    'č': 't͡ʃ',
    'ždž': 'd͡ʒ',
    's': 's',
    'z': 'z',
    'š': 'ʃ',
    'ś': 'ɕ',
    'ž': 'ʒ',
    'x': 'x',
    'r': 'r',
    'ř': 'rʲ',
    'l': 'l',
    'ľ': 'lʲ',
    'v': 'ʋ',
    'j': 'j'
}


def sla_pro_transcribe(s: str) -> str:
    return replace(s, list(proto_slavic_map.items()))


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
def load_data(cog_path: str, swadesh_path: str) -> Tuple[PDF, PDF]:
    # Get cognate data.
    gem_pro = pd.read_csv(cog_path, sep='\t')
    # Get Swadesh list.
    swa = pd.read_csv(swadesh_path, sep='\t', header=None)
    return gem_pro, swa


@run_section('Removing any duplicates or words that do not have a unique reflex...',
             'Removal done.')
def remove_duplicate(cog: PDF, swa: PDF, anc_lang_code: str) -> PDF:
    to_keep = set()
    col = 1 if anc_lang_code == 'la' else 2
    for tokens in swa[col]:
        tokens = tokens.replace('(', '').replace(')', '')
        for token in tokens.split(','):
            to_keep.add(token.strip().strip('*'))
    kept = cog[cog[anc_lang_code].isin(to_keep)].reset_index(drop=True)
    desc = kept[kept['desc_lang'] == lang].reset_index(drop=True)
    dups = {k for k, v in desc[anc_lang_code].value_counts().to_dict().items() if v > 1}
    desc = desc[~desc[anc_lang_code].isin(dups)].reset_index(drop=True)
    return desc


def convert_stress(ipa: str) -> List[str]:
    tokens = i2t(ipa)
    should_stress = False
    ret = list()
    for t in tokens:
        if t.startswith('ˈ') or t.startswith("'"):
            t = t[1:]
            should_stress = True
        elif t.startswith('ˌ'):
            t = t[1:]
        t = str(IPAString(unicode_string=unicodedata.normalize('NFD', t), ignore=True))
        seg = _processor.process(t)
        if isinstance(seg, Nphthong) or (isinstance(seg, Segment) and seg.is_vowel()):
            if should_stress:
                t = t + '{+}'
                should_stress = False
            else:
                t = t + '{-}'
        ret.append(t)
    assert not should_stress
    return ret


def la_transcribe_and_tokenize(text: str, transcriber: LatTranscriber) -> List[str]:
    return convert_stress(transcriber.transcribe(text, with_squared_brackets=False))


if __name__ == "__main__":
    parser = ArgumentParser()
    st.title('Prepare RL dataset.')
    st.header('Specify the arguments first:')
    anc_lang = run_with_argument('ancestor',
                                 parser=parser,
                                 default='',
                                 msg='Ancestor language.')
    anc_lang_code = run_with_argument('ancestor_code',
                                      parser=parser,
                                      default='',
                                      msg='Ancestor language code.')
    lang = run_with_argument('lang',
                             parser=parser,
                             default='',
                             msg='Daughter language.')
    ipa_lang = run_with_argument('ipa_lang',
                                 parser=parser,
                                 default='',
                                 msg='Daughter language in the IPA pickle file.')
    out_dir = run_with_argument('out_dir',
                                parser=parser,
                                default='data/wikt',
                                msg='Output directory')
    cog_path = run_with_argument('cognate_path',
                                 parser=parser,
                                 default='data/gem-pro.tsv',
                                 msg='Path to the cognate data extracted from Wiktionary.')
    swadesh = run_with_argument('swadesh_path',
                                parser=parser,
                                default='data/swadesh_gem_pro.tsv',
                                msg='Path to the Swadesh list.')
    ipa_pickle = run_with_argument('ipa_pickle_path',
                                   parser=parser,
                                   default='data/main.ipa.pkl',
                                   msg='Path to the pickled file that stores all Wiktionary IPA transcriptions.')
    cog, swa = load_data(cog_path, swadesh)
    desc = remove_duplicate(cog, swa, anc_lang_code)
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
        ang_transcriber = AngTranscriber()
        desc[ipa_col] = desc[form_col].apply(lambda s: ang_transcriber.transcribe(
            s.strip('-'), with_squared_brackets=False)).apply(i2t).apply(lambda lst: [replace(x, to_rectify) for x in lst])
    elif lang == 'non':
        ipa_col = 'non_ipa'
        form_col = 'desc_form'
        to_rectify = [('g', 'ɡ'), ('gʷ', 'ɡʷ'), ('h', 'x'), ('hʷ', 'xʷ'), ('ɛ', 'e'), ('ɣ', 'ɡ'), ('ɔ', 'o')]
        non_transcriber = OldNorseTranscription()
        desc[ipa_col] = desc[form_col].apply(lambda s: non_transcriber.transcribe(s).strip('[]')).apply(
            i2t).apply(lambda lst: [replace(x, to_rectify) for x in lst])
    elif lang in ['it', 'es', 'fr', 'uk', 'pl', 'ru']:
        lang2code = {
            'it': 'ita-Latn',
            'es': 'spa-Latn',
            'fr': 'fra-Latn',
            'ru': 'rus-Cyrl',
            'uk': 'ukr-Cyrl',
            'pl': 'pol-Latn'
        }
        transcriber = Epitran(lang2code[lang])
        ipa_col = f'{lang}_ipa'
        form_col = 'desc_form'
        # Italian doesn't have phonemic diphthongs.
        merge_vowels = lang != 'it'
        desc[ipa_col] = desc[form_col].apply(lambda s: i2t(
            transcriber.transliterate(s).replace('ˈ', '').replace('ˌ', '').replace("'", ''), merge_vowels=merge_vowels))
        to_normalize = list()
        if lang == 'ru':
            to_normalize = [('á', 'a'), ('ó', 'o'), ('é', 'e'), ('ú', 'u'),
                            ('ɨ́', 'ɨ'), ('í', 'i'), ('t͡ɕʲ', 't͡ɕ'), ('ʂʲ', 'ʂ')]
        elif lang == 'uk':
            to_normalize = [('ɑ́', 'ɑ'), ('ɔ́', 'ɔ'), ('ɛ́', 'ɛ'), ('í', 'i'), ('ú', 'u'), ('ɪ́', 'ɪ')]
        elif lang == 'pl':
            to_normalize = [('ʐ̇', 'ʐ'), ('t͡ʂ', 'ʈ͡ʂ')]
        elif lang == 'fr':
            to_normalize = [('ù', 'u'), ('â', 'a')]

        def normalize(s):
            for a, b in to_normalize:
                s = s.replace(a, b)
            return s

        desc[ipa_col] = desc[ipa_col].apply(
            lambda lst: [normalize(unicodedata.normalize('NFD', s)) for s in lst])

        # with open(ipa_pickle, 'rb') as fin:
        #     ipa_df = pickle.load(fin)
        # # Use the first IPA transcription if multiple exists, and convert it to a dictionary.
        # ipa_dict = ipa_df[ipa_df['lang'] == ipa_lang].pivot_table(
        #     index='title', values='ipa', aggfunc='first').to_dict()['ipa']
        # ipa_col = f'{lang}_ipa'
        # form_col = 'desc_form'

        # def get_ipa_from_pickle(text: str):
        #     ipa = ipa_dict.get(text, None)
        #     if ipa is not None:
        #         return convert_stress(ipa.strip('/').strip('[]'))
        #     return None

        # desc[ipa_col] = desc[form_col].apply(get_ipa_from_pickle)
        # num_null_entries = pd.isnull(desc[ipa_col]).sum()
        # print(desc[pd.isnull(desc[ipa_col])])
        # assert num_null_entries == 0, num_null_entries
    else:
        raise ValueError(f'Unrecognized language "{lang}".')
    st.write(desc)

    # Get rid of false complex segments.
    show_all_segs(desc[ipa_col])
    desc[ipa_col] = desc[ipa_col].apply(break_false_complex, lang=lang)
    show_all_segs(desc[ipa_col])

    if anc_lang == 'pgmc':
        src_ipa_col = 'pgm_ipa'
        src_form_col = 'gem-pro'
        desc[src_ipa_col] = desc[src_form_col].apply(PGmc_ipa_trans).apply(i2t)
        show_all_segs(desc[src_ipa_col])
        desc[src_ipa_col] = desc[src_ipa_col].apply(break_false_complex, lang='pgm')
        show_all_segs(desc[src_ipa_col])
    elif anc_lang == 'la':
        src_ipa_col = 'la_ipa'
        src_form_col = 'la'
        try:
            transcriber = LatTranscriber(dialect="Classical", reconstruction="Allen")
        except FileNotFoundError:
            lat_fetch = FetchCorpus('lat')
            lat_fetch.import_corpus('lat_models_cltk')
            transcriber = LatTranscriber(dialect="Classical", reconstruction="Allen")

        desc[src_ipa_col] = desc[src_form_col].apply(la_transcribe_and_tokenize, transcriber=transcriber)
        show_all_segs(desc[src_ipa_col])
        desc[src_ipa_col] = desc[src_ipa_col].apply(break_false_complex, lang='la')
        show_all_segs(desc[src_ipa_col])
    elif anc_lang == 'sla-pro':
        src_ipa_col = 'sla_pro_ipa'
        src_form_col = 'sla-pro'
        desc[src_ipa_col] = desc[src_form_col].apply(sla_pro_transcribe).apply(i2t)
        show_all_segs(desc[src_ipa_col])
        desc[src_ipa_col] = desc[src_ipa_col].apply(break_false_complex, lang='sla-pro')
        show_all_segs(desc[src_ipa_col])
    else:
        raise ValueError(f'Unrecognized language "{anc_lang}".')

    src_df = pd.DataFrame()
    src_df['transcription'] = desc[src_form_col]
    src_df['ipa'] = desc[src_ipa_col].apply(''.join)
    src_df['tokens'] = desc[src_ipa_col].apply(' '.join)
    src_df['split'] = 'train'
    data_folder = f'{out_dir}/{anc_lang}-{lang}'
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    src_out_path = f'{data_folder}/{anc_lang}.tsv'
    src_df.to_csv(src_out_path, sep='\t', index=False)
    st.write(f'Source written to {src_out_path}.')

    tgt_df = pd.DataFrame()
    tgt_df['transcription'] = desc[form_col]
    tgt_df['ipa'] = desc[ipa_col].apply(''.join)
    tgt_df['tokens'] = desc[ipa_col].apply(' '.join)
    tgt_df['split'] = 'train'
    tgt_out_path = f'{data_folder}/{lang}.tsv'
    tgt_df.to_csv(tgt_out_path, sep='\t', index=False)
    st.write(f'Target written to {tgt_out_path}.')

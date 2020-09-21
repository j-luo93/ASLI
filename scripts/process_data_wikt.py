import re
import unicodedata
from argparse import ArgumentParser
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pycountry
from cltk.phonology.latin.transcription import Transcriber
from epitran import Epitran
from ipapy.ipastring import IPAString
from lingpy.sequence.sound_classes import ipa2tokens
from loguru import logger


# IPA tokenization including removing leading * (reconstructed terms) and normalizing symbols (done by ipapy).
@lru_cache(maxsize=None)
def i2t(ipa):
    ipa = unicodedata.normalize('NFD', ipa)
    ipa = re.sub(r'^\*', '', ipa)
    tokens = ipa2tokens(ipa, merge_vowels=False, merge_geminates=False)
    ret = list()
    for t in tokens:
        # NOTE(j_luo) Stress symbol is not handled by `ipapy`'s canonicalization process.
        t = t.replace("'", 'ˈ')
        # NOTE(j_luo) Not sure what these symbols mean.
        t = t.replace('̣', '').replace('̧', '').replace('̦', '')
        ret.append(str(IPAString(unicode_string=t)))
    return ret


def read_dict(path: str):
    df = pd.read_csv(path, sep='\t', keep_default_na=False, header=None)
    df.columns = ['lang', 'system', 'grapheme', 'tokens', 'phoneme']
    ret = dict()
    # Use the `tokens` column instead of the `phoneme` column since the former is more detailed.
    # However, a joined str is returned in order to make sure it will be processed by the `i2t`
    # function, for consistency in tokenization.
    for lang, grapheme, tokens in zip(df['lang'], df['grapheme'], df['tokens']):
        ret[(lang, grapheme)] = ''.join(tokens.split())
    return ret


lookup = pycountry.languages.lookup


# Copied from https://stackoverflow.com/questions/48255244/python-check-if-a-string-contains-cyrillic-characters.
def has_cyrillic(text):
    return bool(re.search('[\u0400-\u04FF]', text))


@lru_cache(maxsize=None)
def PGmc_ipa_trans(word: str) -> str:  # only for latin-transliterated Gothic and Greek without diacritics
    # NOTE(j_luo) Based on Frederik's code, with minor modifications.
    word = word.lower()
    word = word.replace('₂', '')
    # vowels
    word = re.sub(r"ē", "eː", word)
    word = re.sub(r"ō", "ɔː", word)
    word = re.sub(r"ā", "aː", word)
    word = re.sub(r"ī", "iː", word)
    word = re.sub(r"ū", "uː", word)

    word = re.sub(r"ô", "ɔːː", word)
    word = re.sub(r"ê", "eːː", word)

    word = re.sub(r'ǭ', 'ɔ̃ː', word)
    word = re.sub(r'ą', 'ã', word)
    word = re.sub(r'į̄', 'ĩː', word)

    # consonants
    word = re.sub(r"h", "x", word)
    word = re.sub(r"f", "ɸ", word)
    word = re.sub(r"xw", "hʷ", word)
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

    word = re.sub(r"g", "ɣ", word)
    word = re.sub(r"ɣw", "gʷ", word)

    word = re.sub(r"nk", "ŋk", word)
    word = re.sub(r"ng", "ŋg", word)
    word = re.sub(r"ng", "ŋg", word)

    return word


G2P_func = Callable[[str], str]  # Grapheme-to-phoneme.
G2T_func = Callable[[str], List[str]]  # Grapheme-to-tokenized phoneme


def get_src_header_and_transcriber(source: str) -> Tuple[str, G2P_func]:
    """Return the column name for the output csv header, and a transcriber for the source language."""
    if source == 'lat':
        try:
            src_transcriber = Transcriber(dialect="Classical", reconstruction="Allen")
            src = 'Latin'
        except FileNotFoundError:
            print("Did not have the corpus `latin_models_cltk`, downloading it now")
            from cltk.corpus.utils.importer import CorpusImporter
            corpus_importer = CorpusImporter('latin')
            corpus_importer.import_corpus('latin_models_cltk')

            src_transcriber = Transcriber(dialect="Classical", reconstruction="Allen")
            src = 'Latin'

        @lru_cache(maxsize=None)
        def src_func(token):
            try:
                ipa = src_transcriber.transcribe(token)
            except IndexError:
                ipa = src_transcriber.transcribe(token, syllabify=False)
            ipa = ipa.strip('[]')
            # Some weird cases of failed macronization.
            ipa = re.sub(r'(.)_', r'\1ː', ipa)
            return ipa
    else:
        src = 'Proto-Germanic'
        src_func = PGmc_ipa_trans
    return src, src_func


def get_tgt_code_and_transcriber(target: str,
                                 pron_dict: Optional[dict] = None,
                                 need_transcriber: bool = True) -> Tuple[str, G2P_func]:
    if target == 'roa-opt':
        tgt_code = 'roa_opt'
    else:
        tgt_code = lookup(target).alpha_3

    if not need_transcriber:
        tgt_g2p = None
    # Use epitran.
    elif pron_dict is None:
        if tgt_code in ['ita', 'spa', 'por', 'fra', 'cat', 'ron', 'deu', 'nld', 'swe']:
            epi_code = f'{tgt_code}-Latn'
        else:
            raise ValueError(f'language {target} not supported.')
        tgt_g2p = Epitran(epi_code).transliterate
    # Use pronunciation dictionary.
    else:
        # Return None if entry not found.
        tgt_g2p = lambda token: pron_dict.get((tgt_code, token), None)

    return tgt_code, tgt_g2p


@dataclass
class Field:
    form: str
    _tokens: List[List[str]] = field(repr=False)  # Raw list of tokens.
    target_side: bool = False
    tokens: str = field(init=False)  # Joined tokens.
    ipa: str = field(init=False)

    def __post_init__(self):
        if self.target_side:
            self.tokens = '|'.join([' '.join(token) for token in self._tokens])
            self.ipa = '|'.join([''.join(token) for token in self._tokens])
        else:
            self.tokens = ' '.join(self._tokens)
            self.ipa = ''.join(self._tokens)

    def to_record(self) -> dict:
        return {
            'transcription': self.form,
            'ipa': self.ipa,
            'tokens': self.tokens
        }


def add_splits(src_df: pd.DataFrame, tgt_df: pd.DataFrame, random_seed: int):
    """Add split column in-place."""
    np.random.seed(random_seed)
    r = np.random.rand(len(src_df))
    splits = list()
    for f in r:
        if f >= 0.8:
            splits.append('test')
        elif f >= 0.7:
            splits.append('dev')
        else:
            splits.append('train')
    src_df['split'] = splits
    tgt_df['split'] = splits


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the Wiktionary cognate data file.')
    parser.add_argument('--source', type=str, help='Source language.')
    parser.add_argument('--targets', type=str, nargs='+', help='Target language.')
    parser.add_argument('--random_seed', type=str, help='Random seed.')
    parser.add_argument('--dict_path', type=str, help='Path to the provided pronunciation dictionary.')
    parser.add_argument('--is_cyrillic', action='store_true', help='Flag to indicate whether it use Cyrillic scripts.')
    parser.add_argument('--no_need_transcriber', dest='need_transcriber', action='store_false',
                        help='Flag to indicate whether transcriber is needed.')
    args = parser.parse_args()

    src_header, src_g2p = get_src_header_and_transcriber(args.source)
    if not args.need_transcriber:
        src_header = args.source
    weird_chars = set("[] #/'")  # Quotation marks in words (not IPA transcriptions) are used for contractions.

    all_df = pd.read_csv(args.data_path, sep='\t', keep_default_na=False)
    pron_dict = read_dict(args.dict_path) if args.dict_path is not None else None

    def gen_pairs(df: pd.DataFrame, src_header: str):
        if args.need_transcriber:
            for src_form, tgt_forms in df.groupby(src_header)['Token']:
                yield src_form, tgt_forms
        else:
            for src_form, tgt_df in df.groupby(src_header)[['Word_Form', 'rawIPA']]:
                yield src_form, tgt_df['Word_Form'].str.strip(), tgt_df['rawIPA'].str.strip()

    lang_col = 'Language' if args.need_transcriber else 'lang_code'
    for target in args.targets:
        df = all_df[all_df[lang_col] == target]

        src_fields: List[Field] = list()
        tgt_fields: List[Field] = list()
        tgt_code, tgt_g2p = get_tgt_code_and_transcriber(target,
                                                         pron_dict=pron_dict,
                                                         need_transcriber=args.need_transcriber)
        for src_form, tgt_forms, *tgt_ipas in gen_pairs(df, src_header):
            if len(src_form) == 0 or len(tgt_forms) == 0:
                continue
            if (set(src_form) & weird_chars) or any(set(t) & weird_chars for t in tgt_forms):
                continue
            # Skip some Cyrillic words.
            if not args.is_cyrillic:
                if has_cyrillic(src_form) or any(has_cyrillic(t) for t in tgt_forms):
                    continue

            if args.need_transcriber:
                tgt_forms = [form for form in tgt_forms if form]

                # Process target side first to skip some pairs due to nonexistent IPA transcriptions (from pronunciation dictionaries).
                tgt_ipas = [tgt_g2p(form) for form in tgt_forms]
                tgt_ipas = [ipa for ipa in tgt_ipas if ipa is not None]
                if not tgt_ipas:
                    continue
            else:
                tgt_ipas = tgt_ipas[0]

            tgt_tokens = [i2t(ipa) for ipa in tgt_ipas]
            tgt_field = Field('|'.join(tgt_forms), tgt_tokens, target_side=True)
            tgt_fields.append(tgt_field)

            src_ipa = src_g2p(src_form)
            src_tokens = i2t(src_ipa)
            src_field = Field(src_form, src_tokens, target_side=False)
            src_fields.append(src_field)
        src_df = pd.DataFrame([field.to_record() for field in src_fields])
        tgt_df = pd.DataFrame([field.to_record() for field in tgt_fields])
        add_splits(src_df, tgt_df, args.random_seed)

        folder = Path(f'./data/wikt/{args.source}-{tgt_code}')
        folder.mkdir(parents=True, exist_ok=True)

        src_path = str(folder / f'{args.source}.tsv')
        src_df.to_csv(src_path, sep='\t', index=None)
        logger.info(f'Data saved to {src_path}.')

        tgt_path = str(folder / f'{tgt_code}.tsv')
        tgt_df.to_csv(tgt_path, sep='\t', index=None)
        logger.info(f'Data saved to {tgt_path}.')

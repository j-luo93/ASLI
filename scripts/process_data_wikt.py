import re
import unicodedata
from argparse import ArgumentParser
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


def get_tgt_code_and_transcriber(target: str, pron_dict: Optional[dict] = None) -> Tuple[str, G2P_func]:
    if target == 'roa-opt':
        tgt_code = 'roa_opt'
    else:
        tgt_code = lookup(target).alpha_3

    # Use epitran.
    if pron_dict is None:
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the Wiktionary cognate data file.')
    parser.add_argument('--source', type=str, help='Source language.')
    parser.add_argument('--targets', type=str, nargs='+', help='Target language.')
    parser.add_argument('--random_seed', type=str, help='Random seed.')
    parser.add_argument('--dict_path', type=str, help='Path to the provided pronunciation dictionary.')
    args = parser.parse_args()

    src_header, src_g2p = get_src_header_and_transcriber(args.source)

    all_df = pd.read_csv(args.data_path, sep='\t', keep_default_na=False)
    pron_dict = read_dict(args.dict_path) if args.dict_path is not None else None
    for target in args.targets:
        tgt_code, tgt_g2p = get_tgt_code_and_transcriber(target, pron_dict=pron_dict)
        df = all_df[all_df['Language'] == target]

        src_cogs = list()  # This stores the transcriptions for src.
        src_ipas = list()  # This stores the phonemes for src.
        src_tokens = list()  # This stores the tokenized phonemes for src.

        tgt_cogs = list()
        tgt_ipas = list()
        tgt_tokens = list()

        weird_chars = set("[] #/'")  # Quotation marks in words (not IPA transcriptions) are used for contractions.
        for src_token, group in df.groupby(src_header)['Token']:
            group = [t for t in group if t]
            if len(src_token) == 0 or len(group) == 0:
                continue
            if (set(src_token) & weird_chars) or any(set(t) & weird_chars for t in group):
                continue
            # Skip some Cyrillic words.
            if has_cyrillic(src_token) or any(has_cyrillic(t) for t in group):
                continue

            # Process target side first to skip some pairs due to nonexistent IPA transcriptions (from pronunciation dictionaries).
            ipas = [tgt_g2p(t) for t in group]
            ipas = [ipa for ipa in ipas if ipa is not None]
            if not ipas:
                continue

            tokens = [i2t(i) for i in ipas]
            tgt_cogs.append('|'.join(group))
            tgt_tokens.append('|'.join([' '.join(token) for token in tokens]))
            tgt_ipas.append('|'.join([''.join(token) for token in tokens]))

            ipa = src_g2p(src_token)
            src_cogs.append(src_token)
            tokens = i2t(ipa)
            src_tokens.append(' '.join(tokens))
            src_ipas.append(''.join(tokens))

        np.random.seed(args.random_seed)
        r = np.random.rand(len(src_cogs))
        splits = list()
        for f in r:
            if f >= 0.8:
                splits.append('test')
            elif f >= 0.7:
                splits.append('dev')
            else:
                splits.append('train')
        src_df = pd.DataFrame({'transcription': src_cogs, 'split': splits, 'ipa': src_ipas, 'tokens': src_tokens})
        tgt_df = pd.DataFrame({'transcription': tgt_cogs, 'split': splits, 'ipa': tgt_ipas, 'tokens': tgt_tokens})

        folder = Path(f'./data/wikt/{args.source}-{tgt_code}')
        folder.mkdir(parents=True, exist_ok=True)

        src_df.to_csv(str(folder / f'{args.source}.tsv'), sep='\t', index=None)
        tgt_df.to_csv(str(folder / f'{tgt_code}.tsv'), sep='\t', index=None)

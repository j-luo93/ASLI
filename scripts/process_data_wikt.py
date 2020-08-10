import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from cltk.phonology.latin.transcription import Transcriber
# from cltk.phonology.old_norse.orthophonology import on as non_trans
from epitran import Epitran
from lingpy.sequence.sound_classes import ipa2tokens

lookup = pycountry.languages.lookup


def PGmc_ipa_trans(word):  # only for latin-transliterated Gothic and Greek without diacritics
    # vowels
    word = re.sub(r"ē", "eː", word)
    word = re.sub(r"ō", "ɔː", word)
    word = re.sub(r"ā", "aː", word)
    word = re.sub(r"ī", "iː", word)
    word = re.sub(r"ū", "uː", word)

    word = re.sub(r"ô", "ɔːː", word)
    word = re.sub(r"ê", "eːː", word)

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the Wiktionary cognate data file.')
    parser.add_argument('--source', type=str, help='Source language.')
    parser.add_argument('--targets', type=str, nargs='+', help='Target language.')
    parser.add_argument('--random_seed', type=str, help='Random seed.')
    args = parser.parse_args()

    for target in args.targets:
        if target == 'roa-opt':
            tgt = 'roa_opt'
        else:
            tgt = lookup(target).alpha_3

        if tgt in ['ita', 'spa', 'por', 'fra', 'cat', 'ron', 'deu', 'nld', 'swe']:
            epi_code = f'{tgt}-Latn'
        else:
            raise ValueError(f'language {target} not supported.')

        if args.source == 'lat':
            src_transcriber = Transcriber(dialect="Classical", reconstruction="Allen")
            src = 'Latin'

            def src_func(token):
                try:
                    ipa = src_transcriber.transcribe(src_token)
                except IndexError:
                    ipa = src_transcriber.transcribe(src_token, syllabify=False)
                ipa = ipa.strip('[]')
                return ipa
        else:
            # src_transcriber = non_trans
            src = 'Proto-Germanic'
            src_func = PGmc_ipa_trans

        tgt_transcriber = Epitran(epi_code)

        np.random.seed(args.random_seed)
        df = pd.read_csv(args.data_path, sep='\t', keep_default_na=False)

        df = df[df['Language'] == target]
        src_cogs = list()
        src_ipas = list()
        src_tokens = list()
        tgt_cogs = list()
        tgt_ipas = list()
        tgt_tokens = list()

        weird_chars = set('[] ')
        for src_token, group in df.groupby(src)['Token']:
            group = [t for t in group if t]
            if len(src_token) == 0 or len(group) == 0:
                continue
            if (set(src_token) & weird_chars) or any(set(t) & weird_chars for t in group):
                continue

            ipa = src_func(src_token)
            src_cogs.append(src_token)
            src_ipas.append(ipa)
            src_tokens.append(' '.join(ipa2tokens(ipa, merge_vowels=False)))

            ipas = [tgt_transcriber.transliterate(t) for t in group]
            tokens = [ipa2tokens(i, merge_vowels=False) for i in ipas]
            tgt_cogs.append('|'.join(group))
            tgt_ipas.append('|'.join(ipas))
            tgt_tokens.append('|'.join([' '.join(token) for token in tokens]))

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

        folder = Path(f'./data/wikt/{args.source}-{tgt}')
        folder.mkdir(parents=True, exist_ok=True)

        src_df.to_csv(str(folder / f'{args.source}.tsv'), sep='\t', index=None)
        tgt_df.to_csv(str(folder / f'{tgt}.tsv'), sep='\t', index=None)

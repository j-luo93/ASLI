from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from cltk.phonology.latin.transcription import Transcriber
from epitran import Epitran
from lingpy.sequence.sound_classes import ipa2tokens

lookup = pycountry.languages.lookup


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the Wiktionary cognate data file.')
    parser.add_argument('--targets', type=str, nargs='+', help='Target language.')
    parser.add_argument('--random_seed', type=str, help='Random seed.')
    args = parser.parse_args()

    for target in args.targets:
        if target == 'roa-opt':
            tgt = 'roa_opt'
        else:
            tgt = lookup(target).alpha_3

        if tgt in ['ita', 'spa', 'por', 'fra', 'cat', 'ron']:
            epi_code = f'{tgt}-Latn'
        else:
            raise ValueError(f'language {target} not supported.')
        lat_transcriber = Transcriber(dialect="Classical", reconstruction="Allen")
        tgt_transcriber = Epitran(epi_code)

        np.random.seed(args.random_seed)
        df = pd.read_csv(args.data_path, sep='\t', keep_default_na=False)

        df = df[df['Language'] == target]
        lat_cogs = list()
        lat_ipas = list()
        lat_tokens = list()
        tgt_cogs = list()
        tgt_ipas = list()
        tgt_tokens = list()

        weird_chars = set('[]')
        for latin, group in df.groupby('Latin')['Token']:
            group = [t for t in group if t]
            if len(latin) == 0 or len(group) == 0:
                continue
            if (set(latin) & weird_chars) or any(set(t) & weird_chars for t in group):
                continue

            lat_cogs.append(latin)
            try:
                ipa = lat_transcriber.transcribe(latin)
            except IndexError:
                ipa = lat_transcriber.transcribe(latin, syllabify=False)
            ipa = ipa.strip('[]')
            lat_ipas.append(ipa)
            lat_tokens.append(' '.join(ipa2tokens(ipa, merge_vowels=False)))

            ipas = [tgt_transcriber.transliterate(t) for t in group]
            tokens = [ipa2tokens(i, merge_vowels=False) for i in ipas]
            tgt_cogs.append('|'.join(group))
            tgt_ipas.append('|'.join(ipas))
            tgt_tokens.append('|'.join([' '.join(token) for token in tokens]))

        r = np.random.rand(len(lat_cogs))
        splits = list()
        for f in r:
            if f >= 0.8:
                splits.append('test')
            elif f >= 0.7:
                splits.append('dev')
            else:
                splits.append('train')
        lat_df = pd.DataFrame({'transcription': lat_cogs, 'split': splits, 'ipa': lat_ipas, 'tokens': lat_tokens})
        tgt_df = pd.DataFrame({'transcription': tgt_cogs, 'split': splits, 'ipa': tgt_ipas, 'tokens': tgt_tokens})

        folder = Path(f'./data/wikt/lat-{tgt}')
        folder.mkdir(parents=True, exist_ok=True)

        lat_df.to_csv(str(folder / 'lat.tsv'), sep='\t', index=None)
        tgt_df.to_csv(str(folder / f'{tgt}.tsv'), sep='\t', index=None)

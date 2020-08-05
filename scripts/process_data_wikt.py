from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

lookup = pycountry.languages.lookup

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the Wiktionary cognate data file.')
    parser.add_argument('--target', type=str, help='Target language.')
    parser.add_argument('--random_seed', type=str, help='Random seed.')
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    df = pd.read_csv(args.data_path, sep='\t', keep_default_na=False)

    df = df[df['Language'] == args.target]
    lat_cogs = list()
    tgt_cogs = list()
    for latin, group in df.groupby('Latin')['Token']:
        if '-' not in latin:
            lat_cogs.append(latin)
            tgt_cogs.append('|'.join(group.tolist()))

    r = np.random.rand(len(lat_cogs))
    splits = list()
    for f in r:
        if f >= 0.8:
            splits.append('test')
        elif f >= 0.7:
            splits.append('dev')
        else:
            splits.append('train')
    lat_df = pd.DataFrame({'cognate': lat_cogs, 'split': splits})
    tgt_df = pd.DataFrame({'cognate': tgt_cogs, 'split': splits})

    if args.target == 'roa-opt':
        tgt = 'roa_opt'
    else:
        tgt = lookup(args.target).alpha_3
    folder = Path(f'./data/wikt/lat-{tgt}')
    folder.mkdir(parents=True, exist_ok=True)

    lat_df.to_csv(str(folder / 'lat.tsv'), sep='\t', index=None)
    tgt_df.to_csv(str(folder / f'{tgt}.tsv'), sep='\t', index=None)

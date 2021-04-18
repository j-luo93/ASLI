import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from sound_law.utils import (load_event, load_stats, read_distance_metrics,
                             read_matching_metrics)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('mode', choices=['run', 'irreg', 'regress', 'merger'], type=str)
    parser.add_argument('prefix', type=str)
    args = parser.parse_args()

    folder = Path(args.folder)

    if args.mode == 'run':
        runs = list(folder.glob('*/'))
        # Get all matching scores.
        match_dfs = list()
        event_dfs = list()
        dist_dfs = list()
        meta_data = list()
        for run in runs:
            saved_dict = {k: v.value for k, v in torch.load(run / 'hparams.pth').items()}
            lang = saved_dict['tgt_lang']
            with open(run / 'best_run', 'r') as fin:
                meta_record = {'best_epoch': int(fin.read(-1)), 'run': str(run)}
                meta_record.update(saved_dict)
                meta_data.append(meta_record)

            # FIXME(j_luo) check we are not missing any entry for the dfs.
            match_df = read_matching_metrics(run).assign(run=str(run))
            event_df = load_event(run).assign(run=str(run))
            dist_df = read_distance_metrics(run).assign(run=str(run))
            match_dfs.append(match_df)
            event_dfs.append(event_df)
            dist_dfs.append(dist_df)

        all_match_df = pd.concat(match_dfs, ignore_index=True)
        all_event_df = pd.concat(event_dfs, ignore_index=True)
        all_dist_df = pd.concat(dist_dfs, ignore_index=True)
        meta_df = pd.DataFrame(meta_data)
        all_match_df.to_csv(f'{args.prefix}_match.tsv', sep='\t', index=False)
        all_event_df.to_csv(f'{args.prefix}_event.tsv', sep='\t', index=False)
        all_dist_df.to_csv(f'{args.prefix}_dist.tsv', sep='\t', index=False)
        meta_df.to_csv(f'{args.prefix}_meta.tsv', sep='\t', index=False)
    else:
        # Get all the data folders based on the mode.
        if args.mode == 'irreg':
            runs = [f'data/wikt/pgmc-rand{i}' for i in range(1, 51)]
        elif args.mode == 'regress':
            runs = [f'data/wikt/pgmc-rand-regress{i}' for i in range(1, 51)]
        else:
            runs = [f'data/wikt/pgmc-rand-merger{i}' for i in range(1, 51)]

        stats_dfs = list()
        for run in runs:
            stats_df = load_stats(run).assign(data_folder=str(run))
            stats_dfs.append(stats_df)
        all_stats_df = pd.concat(stats_dfs, ignore_index=True)
        all_stats_df.to_csv(f'{args.prefix}_stats.tsv', sep='\t', index=False)

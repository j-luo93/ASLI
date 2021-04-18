import re
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator, Tuple

import torch
from tqdm import tqdm


def should_print(out_path_str: str, overwrite: bool) -> bool:
    out_path = Path(out_path_str)
    return not out_path.exists() or out_path.stat().st_size == 0 or overwrite


def get_eval_paths(eval_folder: Path) -> Iterator[Tuple[int, Path]]:
    """Yields tuples of (epoch, path) of saved action sequences (system output) given the folder.

    Args:
        eval_folder (Path): path to the evaluation folder.

    Yields:
        Iterator[Tuple[int, Path]]: an iterator of tuples (epoch, path).
    """
    for eval_path in eval_folder.glob('*.path'):
        match = re.match(r'(\d+).path$', eval_path.name)
        if match is not None:
            epoch = int(match.group(1))
            yield epoch, eval_path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('saved_folder', help='Path to the saved folder.')
    parser.add_argument('mode', choices=['full', 'truncate', 'epoch', 'regress', 'irreg', 'merger', 'evaluate'])
    parser.add_argument('--overwrite', action='store_true', help='Flag to override previous saved files.')
    args = parser.parse_args()

    folder = Path(args.saved_folder)

    runs = list(folder.glob('*/'))
    lang2length = {'got': 20, 'non': 40, 'ang': 60}
    for run in tqdm(runs):
        best_run_path = run / 'best_run'
        with best_run_path.open('r') as fin:
            best_run = int(fin.read(-1).strip())

        saved_dict = torch.load(run / 'hparams.pth')
        tgt_lang = saved_dict['tgt_lang'].value

        if args.mode == 'epoch':
            for epoch, eval_path in get_eval_paths(run / 'eval'):
                base_cmd = f'python sound_law/evaluate/ilp.py --config OPRLPgmc{tgt_lang[0].upper()}{tgt_lang[1:]} --mcts_config SmallSims --cand_path {eval_path} --use_greedy_growth --silent'
                for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    out_path = f'{run}/eval/epoch{epoch}-{m}-100-10.pkl'
                    if should_print(out_path, args.overwrite):
                        print(
                            base_cmd + f' --match_proportion {m} --k_matches 100 --max_power_set_size 10 --out_path {out_path} --message {run}-epoch{epoch}-{m}-100-10')
        elif args.mode in ['full', 'truncate']:
            cand_path = f'{run}/eval/{best_run}.path '
            base_cmd = f'python sound_law/evaluate/ilp.py --config OPRLPgmc{tgt_lang[0].upper()}{tgt_lang[1:]} --mcts_config SmallSims --cand_path {cand_path} --use_greedy_growth --silent'
            if args.mode == 'full':
                # Generate a grid of matching scores.
                for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    for k in [10, 20, 30, 50, 100]:
                        for p in [1, 2, 3, 5, 10]:
                            out_path = f'{run}/eval/full-{m}-{k}-{p}.pkl'
                            if should_print(out_path, args.overwrite):
                                print(
                                    base_cmd + f' --match_proportion {m} --k_matches {k} --max_power_set_size {p} --out_path {out_path} --message {run}-full-{m}-{k}-{p}')

            else:
                # Generate matching scores with different truncate lengths.
                length = lang2length[tgt_lang]
                output = subprocess.run(f'cat {cand_path} | wc -l', shell=True,
                                        text=True, capture_output=True, check=True).stdout
                max_length = int(output)
                for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    for tl in range(5, max_length + 5, 5):
                        pl = min(tl, max_length)
                        out_path = f'{run}/eval/truncate-{m}-100-10-{pl}.pkl'
                        if should_print(out_path, args.overwrite):
                            print(
                                base_cmd + f' --match_proportion {m} --k_matches {100} --max_power_set_size {10} --out_path {out_path} --cand_length {pl} --message {run}-truncate-{m}-100-10-{pl}')
        elif args.mode == 'evaluate':
            # Generate all commands to evaluate paths.
            base_cmd = f'python scripts/evaluate.py --config OPRLPgmc{tgt_lang[0].upper()}{tgt_lang[1:]} --mcts_config SmallSims'
            for epoch, eval_path in get_eval_paths(run / 'eval'):
                in_path = f'{run}/eval/{epoch}.path'
                out_path = in_path + '.scores'
                if should_print(out_path, args.overwrite):
                    print(base_cmd + f' --in_path {in_path} --out_path {out_path} --message {out_path}')
        else:
            base_cmd = f'python sound_law/evaluate/ilp.py --config OPRLPgmcGot --mcts_config SmallSims --in_path data/wikt/pgmc-{tgt_lang}/action_seq.tsv --cand_path {run}/eval/{best_run}.path --use_greedy_growth --silent --tgt_lang {tgt_lang}'
            if args.mode == 'irreg':
                # Generate matching scores for synthetic runs, grouped by number of irregular changes.
                for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    out_path = f'{run}/eval/irreg-{m}-100-10.pkl'
                    if should_print(out_path, args.overwrite):
                        print(
                            base_cmd + f' --match_proportion {m} --k_matches 100 --max_power_set_size 10 --out_path {out_path} --message {run}-irreg-{m}-100-10')
            elif args.mode == 'regress':
                # Generate matching scores for synthetic runs, grouped by number of regressive rules.
                for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    out_path = f'{run}/eval/regress-{m}-100-10.pkl'
                    if should_print(out_path, args.overwrite):
                        print(
                            base_cmd + f' --match_proportion {m} --k_matches 100 --max_power_set_size 10 --out_path {out_path} --message {run}-regress-{m}-100-10')
            else:
                # Generate matching scores for synthetic runs, grouped by number of mergers.
                for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    out_path = f'{run}/eval/merger-{m}-100-10.pkl'
                    if should_print(out_path, args.overwrite):
                        print(
                            base_cmd + f' --match_proportion {m} --k_matches 100 --max_power_set_size 10 --out_path {out_path} --message {run}-merger-{m}-100-10')

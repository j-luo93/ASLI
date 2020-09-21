import math
import re
import subprocess
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from sound_law.config import all_germanic_configs, all_italic_configs
from sound_law.data.cognate import get_paths


def is_metric(tag: str, metric: str) -> bool:
    """Check if `tag` corresponds to the `metric`."""
    name = tag.split('/')[-1]
    return name.startswith(metric)


def is_split(tag: str, split: str) -> bool:
    """Check if `tag` corresponds to the `split`."""
    name = tag.split('/')[1].split('@')[0]
    return name.startswith(split)


strip_pat = re.compile(r'_e$')


def is_lang(item) -> bool:
    """Check if `tag` corresponds to the `lang`."""
    tag, lang = item
    name = strip_pat.sub('', tag.split('/')[1].split('@')[1])
    return name == lang


def get_zs_df(df, src_lang: str, edit_dist: bool = False):
    """Get data frame with the corresponding `src_lang` and metric. If `edit_dist` is True, the metric is set to "edit_dist",
    otherwise it's "precision".
    """
    fam_df = df.query(f'src_lang == "{src_lang}"')
    eval_df = fam_df[fam_df['tag'].str.startswith('eval/')].reset_index(drop=True)

    tag_df = eval_df['tag']
    metric = 'edit_dist' if edit_dist else 'precision'
    mask_m = tag_df.apply(is_metric, metric=metric)
    mask_s = tag_df.apply(is_split, split='test')
    mask_l = eval_df[['tag', 'tgt_lang']].apply(is_lang, axis=1)

    zs_df = eval_df[mask_m & mask_s & mask_l]

    if edit_dist:
        zs_df[['tgt_lang']] = zs_df['tag'].str.extract(r'eval/test@(?P<tgt_lang>\w{3})/edit_dist')
        zs_df['metric'] = 'edit_dist'
    else:
        zs_df[['tgt_lang', 'K']] = zs_df['tag'].str.extract(r'eval/test@(?P<tgt_lang>\w{3})/precision@(?P<K>\d+)$')
        zs_df['metric'] = zs_df['K'].apply(lambda s: 'P@' + s)

    zs_df = zs_df[['step', 'tgt_lang', 'size', 'metric', 'value']]
    return zs_df


def make_line(df, out_name: str, width=1000, height=700):
    chart = alt.Chart(df)

    selection = alt.selection_multi(fields=['tgt_lang'], on='click', bind='legend')

    def round_to(x, up=False, inc=0.05):
        func = math.ceil if up else math.floor
        ret = (func(x / inc) * inc)
        return float(f'{ret:.3f}')

    def encode_y(encoding: str):
        y_min = round_to(df['value'].min())
        y_max = round_to(df['value'].max(), up=True)
        ylim = (y_min, y_max)
        return alt.Y(encoding, scale=alt.Scale(domain=ylim))

    selected_color = alt.condition(selection, 'tgt_lang:N', alt.value('lightgray'))
    selected_opacity = alt.condition(selection, alt.value(1.0), alt.value(0.0))
    line = chart.mark_line().encode(
        x='step:N',
        y=encode_y('mean(value):Q'),
        color=selected_color,
        opacity=selected_opacity,
        tooltip=['step', 'tgt_lang', 'size',
                 alt.Tooltip('max(value)', format='.3f'),
                 alt.Tooltip('min(value)', format='.3f'),
                 alt.Tooltip('mean(value)', format='.3f')]
    ).interactive().properties(width=width,
                               height=height)

    area = chart.mark_area(fillOpacity=0.3).encode(
        x='step:N',
        y=encode_y('max(value):Q'),
        y2='min(value):Q',
        color=selected_color
    ).properties(width=width,
                 height=height)

    ret = (area + line).add_selection(selection)

    out_path = (Path('plots') / out_name).with_suffix('.html')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ret.save(str(out_path))
    return ret


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('tsv_folder', type=str, help='The folder with all tsv files.')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Flag to indicate that the event files are generated in evaluate only mode.')
    args = parser.parse_args()

    tsv_folder = Path(args.tsv_folder)

    if args.evaluate_only:
        # This is for evaluate_only mode.
        name_pat = re.compile(r'ZS(Lat|Pgmc)(\w{3})PhonoNel-run_(\d+)-step_(\d+)$')
    else:
        # This is for the events generated during training.
        name_pat = re.compile(r'ZS(Lat|Pgmc)(\w{3})PhonoNel-run_(\d+)$')

    dfs = dict()
    for tsv_file in tsv_folder.glob('*.tsv'):
        tsv_name = str(tsv_file)
        key = tsv_name.split('__')[2]
        df = pd.read_csv(tsv_name, sep='\t')
        match = name_pat.match(key)
        df['src_lang'] = match.group(1).lower()
        df['tgt_lang'] = match.group(2).lower()
        df['run_id'] = int(match.group(3))
        if args.evaluate_only:
            df['step'] = int(match.group(4))
        dfs[key] = df

    big_df = pd.concat(dfs.values())

    # Obtain dataset sizes.
    sizes = dict()
    for config in chain(all_germanic_configs, all_italic_configs):
        _, tgt_tsv_path = get_paths(config.data_path, config.src_lang, config.tgt_lang)
        out = subprocess.run(f'cat {tgt_tsv_path} | wc -l', shell=True, capture_output=True, encoding='utf8')
        sizes[config.tgt_lang] = int(out.stdout) - 1
    big_df['size'] = big_df['tgt_lang'].apply(sizes.get)

    if args.evaluate_only:
        # For evaluate_only.
        big_df.to_csv('all_processed_eval.tsv', sep='\t', index=None)
    else:
        # For training.
        big_df.to_csv('all_processed.tsv', sep='\t', index=None)

    # Obtain family results.
    # Precision results.
    zs_italic_df = get_zs_df(big_df, 'lat')
    zs_germanic_df = get_zs_df(big_df, 'pgmc')

    zs_italic_p1 = zs_italic_df.query('metric == "P@1"')
    zs_italic_p5 = zs_italic_df.query('metric == "P@5"')
    zs_germanic_p1 = zs_germanic_df.query('metric == "P@1"')
    zs_germanic_p5 = zs_germanic_df.query('metric == "P@5"')

    make_line(zs_italic_p1, 'italic_p1')
    make_line(zs_italic_p5, 'italic_p5')
    make_line(zs_germanic_p1, 'germanic_p1')
    make_line(zs_germanic_p5, 'germanic_p5')
    # Edit distance results.
    zs_italic_edist = get_zs_df(big_df, 'lat', edit_dist=True)
    zs_germanic_edist = get_zs_df(big_df, 'pgmc', edit_dist=True)

    make_line(zs_italic_edist, 'italic_edist')
    make_line(zs_germanic_edist, 'germanic_edist')

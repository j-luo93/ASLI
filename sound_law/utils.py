import pickle
import re
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union

import pandas as pd
import streamlit as st
import torch
from google.protobuf.json_format import MessageToDict
from numpy.core.function_base import _add_docstring
from pypheature.nphthong import Nphthong
from pypheature.process import FeatureProcessor
from pypheature.segment import Segment
from sklearn.metrics import auc
from tensorflow.python.summary.summary_iterator import summary_iterator

PDF = pd.DataFrame

PathLike = Union[Path, str]


def run_section(before_msg: str, after_msg: str, **kwargs):

    def decorator(func):

        cached_func = st.cache(hash_funcs={PDF: id, Segment: id, Nphthong: id, FeatureProcessor: id}, **kwargs)(func)

        @wraps(cached_func)
        def wrapped(*args, **kwargs):
            status_text = st.subheader(before_msg)
            ret = cached_func(*args, **kwargs)
            status_text.subheader(before_msg + '\t' + after_msg)
            return ret

        return wrapped

    return decorator


def run_with_argument(name: str, *, parser: Optional[ArgumentParser] = None, default: Optional[Any] = None, msg: Optional[str] = None):
    if st._is_running_with_streamlit:
        argument = st.text_input(f'{name}:', default, help=msg)
        return argument
    else:
        assert parser is not None, 'Must pass parser for script mode.'
        parser.add_argument(f'--{name}', default=default, help=msg)
        args = parser.parse_known_args()[0]
        return getattr(args, name)


def read_matching_score(pickle_path: PathLike) -> float:
    """Reads the matching score from a saved pickle file.

    Args:
        pickle_path (PathLike): path to the saved pickle file.

    Returns:
        float: the matching score. Defaults to -1 if `pickle_path` doesn't exists.
    """
    pickle_path = Path(pickle_path)
    try:
        with pickle_path.open('rb') as fin:
            saved_results = pickle.load(fin)
            matching, status, final_value, total_null_costs, size_cnt = saved_results
            assert status == 0
            return 1.0 - final_value / total_null_costs
    except FileNotFoundError:
        return -1


def read_matching_metrics(folder_path: PathLike) -> PDF:
    """Reads all matching metrics saved in pickle files, and returns a `DataFrame` that summarizes all results.

    Args:
        folder_path (PathLike): path to the folder that stores all pickle files.

    Returns:
        pd.DataFrame: the dataframe that summarizes all metrics. It contains the following columns:
            match_name, match_proportion, k_matches, max_power_set_size, truncate_length and match_score.
    """
    folder_path = Path(folder_path)
    # Pattern used for the majority of all matching results.
    match_pat1 = re.compile(r'(?P<name>\w+)-(?P<mp>\d+)-(?P<km>\d+)-(?P<mpss>\d+)(-(?P<tl>\d+))?.pkl')
    # Pattern used for matching results with epoch number.
    match_pat2 = re.compile(r'epoch(?P<epoch>\d+)-(?P<mp>\d+)-(?P<km>\d+)-(?P<mpss>\d+).pkl')

    records = list()
    for file_path in folder_path.glob('*.pkl'):
        match = match_pat1.match(file_path.name)
        if match is None:
            matched_pat = 1
        else:
            match = match_pat2.match(file_path.name)
            matched_pat = 2

        if match is None:
            continue

        name = match.group('name') if matched_pat == 1 else 'epoch'
        mp = match.group('mp')
        km = match.group('km')
        mpss = match.group('mpss')
        tl = match.group('tl') if matched_pat == 1 else None
        epoch = match.group('epoch') if matched_pat == 2 else None
        score = read_matching_score(file_path)
        records.append({'match_name': name,
                        'match_proportion': mp,
                        'k_matches': km,
                        'max_power_set_size': mpss,
                        'truncate_length': tl,
                        'epoch': epoch,
                        'match_score': score})
    match_df = pd.DataFrame(records)
    return match_df


@dataclass
class Record:
    """Represents one single record from the event file.
    """
    wall_time: float
    tag: str
    value: float
    step: Optional[int] = None


class EventFile:
    """This is a wrapper class to access the records stored in an event file (produced by tensorboard)."""

    def __init__(self, path: PathLike):
        self.path = Path(path)

    def __iter__(self) -> Iterator[Record]:
        """`summary_iterator` yields a structured record that can be accessed by first calling `MessageToDict`.
        Afterwards, it can be accessed like a normal dict with a structure as follows:

        wallTime: float
        (optional) fileVersion: str
        (optional) step: int
        (optional) summary:
            value: [
                tag: str
                simpleValue: float
            ]
        Brackets mean it can have multiple values (like a list).
        """
        default_step = Counter()
        for e in summary_iterator(str(self.path)):
            e = MessageToDict(e)
            wall_time = e['wallTime']
            try:
                v = e['summary']['value']
                assert len(v) == 1
                v = v[0]
                tag = v['tag']
                value = float(v['simpleValue'])
                if abs(value - 2.0) < 1e-6 and tag == 'best_score':
                    value = 1.0
                try:
                    step = int(e['step'])
                except KeyError:
                    step = default_step[tag]
                    default_step[tag] += 1
                yield Record(wall_time, tag, value, step=step)
            except KeyError:
                pass


def load_event(run_folder: PathLike) -> PDF:
    """Loads the event file and returns a dataframe summarizing the results.

    Args:
        run_folder (PathLike): path to the folder with the run.

    Returns:
        pd.DataFrame: the dataframe summarizing the result. It includes the columns corresponding to `Record`'s fields.
    """
    run_folder = Path(run_folder)
    records = list()
    event_file = EventFile(list(Path(run_folder).glob('events*'))[0])
    for record in event_file:
        record = asdict(record)
        records.append(record)
    record_df = pd.DataFrame(records)
    return record_df


def read_distance_metrics(run_folder: PathLike) -> PDF:
    """Reads all distance scores from `run_folder`. Returns a dataframe summarizing all results.

    Args:
        run_folder (PathLike): path to the saved folder with the run

    Returns:
        PDF: a dataframe summarizing all results. It includes columns:
            truncate_length, epoch, and distance_score.
    """
    run_folder = Path(run_folder)
    records = list()
    for score_path in run_folder.glob('eval/*.path.scores'):
        epoch = re.match(r'(\w+).path.scores', score_path.name).group(1)
        with open(score_path) as fin:
            truncated_dists = list()
            for line in fin:
                truncated_dists.append(float(line))
        start_dist = truncated_dists[0]
        for dist, l in enumerate(truncated_dists[1:], 1):
            records.append({'truncate_length': l,
                            'epoch': epoch,
                            'distance_score': 1.0 - dist / start_dist})
    record_df = pd.DataFrame(records)
    return record_df


def load_stats(data_folder: PathLike) -> PDF:
    """Loads stats of the action sequence in the data folder. Returns a dataframe summarizing the results.

    Args:
        data_folder (PathLike): path to the data folder.

    Returns:
        PDF: a dataframe summarizing the stats. It includes n_merger, n_split, n_loss, n_irreg and n_regress.
    """
    data_folder = Path(data_folder)
    action_seq_df = pd.read_csv(data_folder / 'action_seq.tsv', sep='\t')
    n_merger = action_seq_df['is_merger_bool'].sum()
    n_split = action_seq_df['is_split_bool'].sum()
    n_loss = action_seq_df['is_loss_bool'].sum()
    n_irreg = (action_seq_df['num_aff'] < 2).sum()
    try:
        # Some older data folders do not have this column.
        n_regress = action_seq_df['is_regressive'].sum()
    except KeyError:
        n_regress = None
    records = {'n_merger': n_merger, 'n_split': n_split, 'n_loss': n_loss, 'n_irreg': n_irreg, 'n_regress': n_regress}
    return pd.DataFrame([records])

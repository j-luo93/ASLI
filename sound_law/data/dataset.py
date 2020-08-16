from __future__ import annotations

import logging
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union, overload

import numpy as np
import pandas as pd
import torch
from panphon.featuretable import FeatureTable
from torch.utils.data import Dataset

from dev_misc import LT, add_argument, g
from dev_misc.devlib.helper import get_array
from dev_misc.utils import handle_sequence_inputs, cached_property

_ft = FeatureTable()


SOT = '<SOT>'
EOT = '<EOT>'
PAD = '<pad>'
SOT_ID = 0
EOT_ID = 1
PAD_ID = 2


DF = pd.DataFrame

add_argument('use_stress', dtype=bool, default=True, msg='Flag to use stress.')
add_argument('use_duration', dtype=bool, default=True, msg='Flag to use duration (long or short).')
add_argument('use_diacritics', dtype=bool, default=True, msg='Flag to use diacritics.')


@handle_sequence_inputs
def _preprocess(s: str) -> str:
    s = unicodedata.normalize('NFD', s)

    def one_pass(s):
        if not g.use_stress and s[0] == "ˈ":
            s = s[1:]
        if not g.use_duration and s[-1] == 'ː':
            s = s[:-1]
        if not g.use_diacritics and unicodedata.category(s[-1]) in ['Mn', 'Lm']:
            s = s[:-1]
        return s

    while True:
        new_s = one_pass(s)
        if s == new_s:
            break
        s = new_s
    return s


def _get_contents(df: DF, input_format: str) -> Tuple[List[List[str]], List[str]]:
    if input_format == 'wikt':
        contents = list()
        sources = list()
        for seqs, src in zip(df['tokens'], df['source']):
            for tokens in seqs.split('|'):
                contents.append(_preprocess(tokens.split()))
                sources.append(src)
    else:
        contents = [_preprocess(tokens) for tokens in df['tokens'].str.split()]
        sources = df['source'].tolist()
    return contents, sources


class Alphabet:
    """A class to represent the alphabet of any dataset."""

    def __init__(self, lang: str, contents: List[List[str]], sources: Optional[Union[str, List[str]]] = None):
        if sources is not None:
            if isinstance(sources, str):
                sources = [sources] * len(contents)
            else:
                assert len(contents) == len(sources)
        else:
            sources = ['unknown'] * len(contents)

        cnt = defaultdict(Counter)
        for content, source in zip(contents, sources):
            for c in content:
                cnt[c][source] += 1
        units = sorted(cnt.keys())
        self.special_units = [SOT, EOT]
        self.special_ids = [SOT_ID, EOT_ID]
        self._id2unit = self.special_units + units
        self._unit2id = dict(zip(self.special_units, self.special_ids))
        self._unit2id.update({c: i for i, c in enumerate(units, len(self.special_units))})
        self.stats: pd.DataFrame = pd.DataFrame.from_dict(cnt)

        logging.info(f'Alphabet for {lang}, size {len(self._id2unit)}: {self._id2unit}.')

    @property
    def pfm(self) -> LT:
        """Phonological feature matrix for the entire alphabet. For the special units, use all 0's."""
        pfvs = [torch.zeros(22).long() for _ in range(len(self.special_ids))]
        for unit in self._id2unit[len(self.special_ids):]:
            pfv = self.get_pfv(unit)
            pfvs.append(pfv)
        pfm = torch.stack(pfvs, dim=0).refine_names(..., 'phono_feat')
        return pfm

    def get_pfv(self, s: str) -> LT:
        """Get phonological feature vector (pfv) for a unit."""
        ret = _ft.word_to_vector_list(s, numeric=True)
        if len(ret) != 1:
            raise ValueError(f'Inconsistent tokenization results between panphon and lingpy.')

        # NOTE(j_luo) `+1` since the original features range from -1 to 1.
        ret = torch.LongTensor(ret[0]) + 1
        return ret

    @classmethod
    def from_tsv(cls, lang: str, path: str, input_format: str) -> Alphabet:
        df = pd.read_csv(path, sep='\t')
        df['source'] = path
        contents, sources = _get_contents(df, input_format)
        return cls(lang, contents, sources=sources)

    @classmethod
    def from_tsvs(cls, lang: str, paths: List[str], input_format: str) -> Alphabet:
        dfs = list()
        for path in paths:
            df = pd.read_csv(path, sep='\t')
            df['source'] = path
            dfs.append(df)
        df = pd.concat(dfs)
        contents, sources = _get_contents(df, input_format)
        return cls(lang, contents, sources=sources)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, unit: str) -> int: ...

    def __getitem__(self, index_or_unit):
        if isinstance(index_or_unit, int):
            return self._id2unit[index_or_unit]
        elif isinstance(index_or_unit, str):
            return self._unit2id[index_or_unit]
        else:
            raise TypeError(f'Unsupported type for "{index_or_unit}".')

    def __len__(self):
        return len(self._unit2id)


@dataclass
class Split:
    """A class representing a split configuration."""
    main_split: str
    folds: List[int] = None

    def __post_init__(self):
        assert self.main_split in ['train', 'dev', 'test', 'all']

    def select(self, df: DF) -> DF:
        if self.main_split == 'all':
            return df

        ret = df.copy()
        if self.folds is None:
            values = {self.main_split}
        else:
            values = {str(f) for f in self.folds}
        return ret[ret['split'].isin(values)].reset_index(drop=True)


def get_paths(data_path: Path, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
    prefix = data_path / f'{src_lang}-{tgt_lang}'
    src_path = f'{prefix / src_lang}.tsv'
    tgt_path = f'{prefix / tgt_lang}.tsv'
    return src_path, tgt_path


class OnePairDataset(Dataset):

    def __init__(self,
                 data_path: Path,
                 split: Split,
                 src_lang: str,
                 tgt_lang: str,
                 src_abc: Alphabet,
                 tgt_abc: Alphabet,
                 input_format: str,
                 keep_ratio: Optional[float] = None):
        self.split = split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        src_path, tgt_path = get_paths(data_path, src_lang, tgt_lang)

        src_df = pd.read_csv(str(src_path), sep='\t')
        tgt_df = pd.read_csv(str(tgt_path), sep='\t')

        if input_format == 'wikt':
            src_df = src_df.rename(columns={col: f'src_{col}' for col in src_df.columns})
            tgt_df = tgt_df.rename(columns={col: f'tgt_{col}' for col in tgt_df.columns})
            cat_df: pd.DataFrame = pd.concat([src_df, tgt_df], axis=1)
            cat_df['split_tgt_tokens'] = cat_df['tgt_tokens'].str.split('|')
            cat_df['num_variants'] = cat_df['split_tgt_tokens'].apply(len)
            cat_df['sample_weight'] = 1.0 / cat_df['num_variants']
            cat_df = cat_df.explode('split_tgt_tokens')

            src_df = cat_df[['src_split', 'src_tokens', 'sample_weight']]
            tgt_df = cat_df[['tgt_split', 'split_tgt_tokens', 'sample_weight']]

            src_df = src_df.rename(columns={'src_split': 'split', 'src_tokens': 'tokens'})
            tgt_df = tgt_df.rename(columns={'tgt_split': 'split', 'split_tgt_tokens': 'tokens'})

        src_df = self.split.select(src_df)
        tgt_df = self.split.select(tgt_df)

        if keep_ratio is not None:
            logging.info(f'keep_ratio is {keep_ratio}.')
            num = int(len(src_df) * keep_ratio)
            src_df = src_df.loc[:num]
            tgt_df = tgt_df.loc[:num]

        token_col = 'tokens' if input_format == 'wikt' else 'parsed_tokens'
        self.src_unit_seqs = get_array([_preprocess(tokens) for tokens in src_df[token_col].str.split()])
        self.tgt_unit_seqs = get_array([_preprocess(tokens) for tokens in tgt_df[token_col].str.split()])

        self.src_vocab = np.asarray([''.join(us) for us in self.src_unit_seqs])
        self.tgt_vocab = np.asarray([''.join(us) for us in self.tgt_unit_seqs])

        self.src_id_seqs = [[src_abc[u] for u in seq] for seq in self.src_unit_seqs]
        self.tgt_id_seqs = [[tgt_abc[u] for u in seq] for seq in self.tgt_unit_seqs]

        logging.info(f'Total number of cognate pairs for {src_lang}-{tgt_lang} for {split}: {len(self.src_vocab)}.')

        self.sample_weights: Optional[np.ndarray] = None
        if input_format == 'wikt':
            self.sample_weights = src_df['sample_weight'].values

    def __getitem__(self, index: int):
        return {
            'src_id_seq': [SOT_ID] + self.src_id_seqs[index] + [EOT_ID],
            'src_unit_seq': [SOT] + self.src_unit_seqs[index] + [EOT],
            'tgt_id_seq': self.tgt_id_seqs[index] + [EOT_ID],
            'tgt_unit_seq': self.tgt_unit_seqs[index] + [EOT],
            'index': index,
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang
        }

    def __len__(self):
        return len(self.src_id_seqs)

    @cached_property
    def max_seq_length(self) -> int:
        '''Returns the max sequence length among sequences in this Dataset'''
        return max(map(len, self.src_unit_seqs)) + 2 # the +2 comes from the SOT and EOT tokens

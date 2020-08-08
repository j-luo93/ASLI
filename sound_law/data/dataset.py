from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Union, overload

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from dev_misc.devlib.helper import get_array

SOT = '<SOT>'
EOT = '<EOT>'
SOT_ID = 0
EOT_ID = 1


DF = pd.DataFrame


def _get_contents(df: DF, input_format: str) -> List[List[str]]:
    if input_format == 'wikt':
        contents = list()
        for seqs in df['tokens']:
            for tokens in seqs.split('|'):
                contents.append(tokens.split())
    else:
        contents = df['tokens'].str.split().tolist()
    return contents


class Alphabet:
    """A class to represent the alphabet of any dataset."""

    def __init__(self, lang: str, contents: List[List[str]]):
        data = set()
        for content in contents:
            data.update(content)
        data = sorted(data)
        special_units = [SOT, EOT]
        self._id2unit = special_units + data
        self._unit2id = {SOT: SOT_ID, EOT: EOT_ID}
        self._unit2id.update({c: i for i, c in enumerate(data, len(special_units))})

        logging.info(f'Alphabet for {lang}, size {len(self._id2unit)}: {self._id2unit}.')

    @classmethod
    def from_tsv(cls, lang: str, path: str, input_format: str) -> Alphabet:
        df = pd.read_csv(path, sep='\t')
        contents = _get_contents(df, input_format)
        return cls(lang, contents)

    @classmethod
    def from_tsvs(cls, lang: str, paths: List[str], input_format: str) -> Alphabet:
        dfs = list()
        for path in paths:
            df = pd.read_csv(path, sep='\t')
            dfs.append(df)
        df = pd.concat(dfs)
        contents = _get_contents(df, input_format)
        return cls(lang, contents)

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
                 input_format: str):
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

            src_df = cat_df[['src_ipa', 'src_split', 'src_tokens', 'sample_weight']]
            tgt_df = cat_df[['tgt_ipa', 'tgt_split', 'split_tgt_tokens', 'sample_weight']]

            src_df = src_df.rename(columns={'src_ipa': 'transcription', 'src_split': 'split', 'src_tokens': 'tokens'})
            # FIXME(j_luo) Transcription column contains '|' here.
            tgt_df = tgt_df.rename(columns={'tgt_ipa': 'transcription',
                                            'tgt_split': 'split', 'split_tgt_tokens': 'tokens'})

        src_df = self.split.select(src_df)
        tgt_df = self.split.select(tgt_df)

        self.src_vocab = get_array(src_df['transcription'])
        self.tgt_vocab = get_array(tgt_df['transcription'])

        token_col = 'tokens' if input_format == 'wikt' else 'parsed_tokens'
        self.src_unit_seqs = get_array(src_df[token_col].str.split().to_list())
        self.tgt_unit_seqs = get_array(tgt_df[token_col].str.split().to_list())

        self.src_id_seqs = [[src_abc[u] for u in seq] for seq in self.src_unit_seqs]
        self.tgt_id_seqs = [[tgt_abc[u] for u in seq] for seq in self.tgt_unit_seqs]

        logging.info(f'Total number of cognate pairs for {src_lang}-{tgt_lang} for {split}: {len(self.src_vocab)}.')

        self.sample_weights: Optional[np.ndarray] = None
        if input_format == 'wikt':
            self.sample_weights = src_df['sample_weight'].values

    def __getitem__(self, index: int):
        return {
            'src_id_seq': self.src_id_seqs[index],
            'src_unit_seq': self.src_unit_seqs[index],
            'tgt_id_seq': self.tgt_id_seqs[index] + [EOT_ID],
            'tgt_unit_seq': self.tgt_unit_seqs[index] + [EOT],
            'index': index,
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang
        }

    def __len__(self):
        return len(self.src_id_seqs)

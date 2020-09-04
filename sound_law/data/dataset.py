from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union, overload

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dev_misc import LT, NDA, Arg, add_argument, add_check, g
from dev_misc.devlib.helper import get_array
from dev_misc.utils import cached_property, handle_sequence_inputs

from .alphabet import EOT, EOT_ID, SOT, SOT_ID

DF = pd.DataFrame


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
        # NOTE(j_luo) Do not call `reset_index` here.
        return ret[ret['split'].isin(values)]


class OnePairDataset(Dataset):

    # def __init__(self,
    #              data_path: Path,
    #              split: Split,
    #              src_lang: str,
    #              tgt_lang: str,
    #              src_abc: Alphabet,
    #              tgt_abc: Alphabet,
    #              input_format: str,
    #              keep_ratio: Optional[float] = None):
    #     self.split = split
    #     self.src_lang = src_lang
    #     self.tgt_lang = tgt_lang

    #     src_path, tgt_path = get_paths(data_path, src_lang, tgt_lang)

    #     src_df = pd.read_csv(str(src_path), sep='\t')
    #     tgt_df = pd.read_csv(str(tgt_path), sep='\t')

    #     if input_format == 'wikt':
    #         src_df = src_df.rename(columns={col: f'src_{col}' for col in src_df.columns})
    #         tgt_df = tgt_df.rename(columns={col: f'tgt_{col}' for col in tgt_df.columns})
    #         cat_df: pd.DataFrame = pd.concat([src_df, tgt_df], axis=1)
    #         cat_df['split_tgt_tokens'] = cat_df['tgt_tokens'].str.split('|')
    #         cat_df['num_variants'] = cat_df['split_tgt_tokens'].apply(len)
    #         cat_df['sample_weight'] = 1.0 / cat_df['num_variants']
    #         cat_df = cat_df.explode('split_tgt_tokens')

    #         src_df = cat_df[['src_split', 'src_tokens', 'sample_weight']]
    #         tgt_df = cat_df[['tgt_split', 'split_tgt_tokens', 'sample_weight']]

    #         src_df = src_df.rename(columns={'src_split': 'split', 'src_tokens': 'tokens'})
    #         tgt_df = tgt_df.rename(columns={'tgt_split': 'split', 'split_tgt_tokens': 'tokens'})

    #     src_df = self.split.select(src_df)
    #     tgt_df = self.split.select(tgt_df)

    #     if keep_ratio is not None:
    #         logging.imp(f'keep_ratio is {keep_ratio}. Note that this is not randomized.')
    #         num = int(len(src_df) * keep_ratio)
    #         src_df = src_df.loc[:num]
    #         tgt_df = tgt_df.loc[:num]

    #     token_col = 'tokens' if input_format == 'wikt' else 'parsed_tokens'

    #     def get_unit_seqs(df, abc: Alphabet):
    #         # TODO(j_luo) A bit hacky here.
    #         std_func = handle_sequence_inputs(lambda s: abc.standardize(s))
    #         return get_array([std_func(_preprocess(tokens)) for tokens in df[token_col].str.split()])

    #     self.src_unit_seqs = get_unit_seqs(src_df, src_abc)
    #     self.tgt_unit_seqs = get_unit_seqs(tgt_df, tgt_abc)

    #     self.src_vocab = np.asarray([''.join(us) for us in self.src_unit_seqs])
    #     self.tgt_vocab = np.asarray([''.join(us) for us in self.tgt_unit_seqs])

    #     self.src_id_seqs = [[src_abc[u] for u in seq] for seq in self.src_unit_seqs]
    #     self.tgt_id_seqs = [[tgt_abc[u] for u in seq] for seq in self.tgt_unit_seqs]

    #     logging.info(f'Total number of cognate pairs for {src_lang}-{tgt_lang} for {split}: {len(self.src_vocab)}.')

    #     self.sample_weights: Optional[np.ndarray] = None
    #     if input_format == 'wikt':
    #         self.sample_weights = src_df['sample_weight'].values
    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 src_unit_seqs: NDA,
                 tgt_unit_seqs: NDA,
                 src_id_seqs: NDA,
                 tgt_id_seqs: NDA,
                 src_vocab: NDA,
                 tgt_vocab: NDA,
                 sample_weights: NDA):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_unit_seqs = src_unit_seqs
        self.tgt_unit_seqs = tgt_unit_seqs
        self.src_id_seqs = src_id_seqs
        self.tgt_id_seqs = tgt_id_seqs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.sample_weights = sample_weights

    @classmethod
    def from_dataframe(cls, src_lang: str, tgt_lang: str, df: DF) -> OnePairDataset:
        return cls(src_lang, tgt_lang,
                   df['post_unit_seq_src'].values,
                   df['post_unit_seq_tgt'].values,
                   df['id_seq_src'].values,
                   df['id_seq_tgt'].values,
                   df['form_src'].values,
                   df['form_tgt'].values,
                   df['sample_weight'].values)

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
        return max(map(len, self.src_unit_seqs)) + 2  # the +2 comes from the SOT and EOT tokens

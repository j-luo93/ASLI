from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple, Union, overload

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


class Vocabulary:
    """This is different from a dataset in that every form can occur exactly once."""

    def __init__(self, forms: NDA, unit_seqs: NDA, id_seqs: NDA):
        df = pd.DataFrame({'form': forms, 'unit_seq': unit_seqs, 'id_seq': id_seqs})
        mask = df.duplicated(subset='form')
        df = df[mask]
        self.forms = df['form'].values
        self.unit_seqs = df['unit_seq'].values
        self.id_seqs = df['id_seq'].values

    def __getitem__(self, idx: int) -> dict:
        return {
            'form': self.forms[idx],
            'unit_seq': self.unit_seqs[idx],
            'id_seq': self.id_seqs[idx]
        }

    def __len__(self):
        return len(self.forms)


class OnePairDataset(Dataset):

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 src_unit_seqs: NDA,
                 tgt_unit_seqs: NDA,
                 src_id_seqs: NDA,
                 tgt_id_seqs: NDA,
                 src_forms: NDA,
                 tgt_forms: NDA,
                 sample_weights: NDA):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_unit_seqs = src_unit_seqs
        self.tgt_unit_seqs = tgt_unit_seqs
        self.src_id_seqs = src_id_seqs
        self.tgt_id_seqs = tgt_id_seqs
        self.src_forms = src_forms
        self.tgt_forms = tgt_forms
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
            'tgt_lang': self.tgt_lang,
            'src_form': self.src_forms[index],
            'tgt_form': self.tgt_forms[index],
        }

    def __len__(self):
        return len(self.src_id_seqs)

    @cached_property
    def max_seq_length(self) -> int:
        '''Returns the max sequence length among sequences in this Dataset'''
        return max(map(len, self.src_unit_seqs)) + 2  # the +2 comes from the SOT and EOT tokens

    @cached_property
    def src_vocabulary(self) -> Vocabulary:
        return Vocabulary(self.src_forms, self.src_unit_seqs, self.src_id_seqs)

    @cached_property
    def tgt_vocabulary(self) -> Vocabulary:
        return Vocabulary(self.tgt_forms, self.tgt_unit_seqs, self.tgt_id_seqs)

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
from .setting import Setting

DF = pd.DataFrame


def pad(seq: list, sot: bool, eot: bool, raw: str) -> list:
    before = [SOT if raw else SOT_ID] * sot
    after = [EOT if raw else EOT_ID] * eot
    return before + seq + after


class Vocabulary:
    """This is different from a dataset in that every form can occur exactly once."""

    def __init__(self, forms: NDA, unit_seqs: NDA, id_seqs: NDA, sot: bool, eot: bool):
        df = pd.DataFrame({'form': forms, 'unit_seq': unit_seqs, 'id_seq': id_seqs})
        mask = df.duplicated(subset='form')
        df = df[~mask]
        self.forms = df['form'].values
        self._form2id = {form: i for i, form in enumerate(self.forms)}
        self.unit_seqs = df['unit_seq'].values
        self.id_seqs = df['id_seq'].values
        self.sot = sot
        self.eot = eot

    def __getitem__(self, idx: int) -> dict:
        return {
            'form': self.forms[idx],
            'unit_seq': pad(self.unit_seqs[idx], self.sot, self.eot, True),
            'id_seq': pad(self.id_seqs[idx], self.sot, self.eot, False)
        }

    def get_id_by_form(self, form: str) -> int:
        return self._form2id[form]

    def __len__(self):
        return len(self.forms)


class OnePairDataset(Dataset):

    def __init__(self,
                 setting: Setting,
                 df: DF):
        self.setting = setting
        self.src_unit_seqs = df['post_unit_seq_src'].values
        self.tgt_unit_seqs = df['post_unit_seq_tgt'].values
        self.src_id_seqs = df['id_seq_src'].values
        self.tgt_id_seqs = df['id_seq_tgt'].values
        self.src_forms = df['form_src'].values
        self.tgt_forms = df['form_tgt'].values
        self.sample_weights = df['sample_weight'].values

    def __getitem__(self, index: int):
        return {
            'src_id_seq': pad(self.src_id_seqs[index],
                              self.setting.src_sot,
                              self.setting.src_eot, False),
            'src_unit_seq': pad(self.src_unit_seqs[index],
                                self.setting.src_sot,
                                self.setting.src_eot, True),
            'tgt_id_seq': pad(self.tgt_id_seqs[index],
                              self.setting.tgt_sot,
                              self.setting.tgt_eot, False),
            'tgt_unit_seq': pad(self.tgt_unit_seqs[index],
                                self.setting.tgt_sot,
                                self.setting.tgt_eot, True),
            'index': index,
            'src_lang': self.setting.src_lang,
            'tgt_lang': self.setting.tgt_lang,
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
        return Vocabulary(self.src_forms, self.src_unit_seqs, self.src_id_seqs, self.setting.src_sot, self.setting.src_eot)

    @cached_property
    def tgt_vocabulary(self) -> Vocabulary:
        return Vocabulary(self.tgt_forms, self.tgt_unit_seqs, self.tgt_id_seqs, self.setting.tgt_sot, self.setting.tgt_eot)

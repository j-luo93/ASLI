from .dataset import Alphabet
from dataclasses import field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

from dev_misc import BT, LT, NDA, add_argument, g
from dev_misc.devlib import BaseBatch, batch_class, pad_to_dense
from dev_misc.devlib.helper import get_array, get_tensor, has_gpus
from dev_misc.trainlib import Task
from dev_misc.trainlib.base_data_loader import (BaseDataLoader,
                                                BaseDataLoaderRegistry)
from dev_misc.utils import cached_property
from sound_law.data.dataset import OnePairDataset

from .dataset import Split


@batch_class
class PaddedUnitSeqs(BaseBatch):
    """`unit_seqs` should not be transposed, but the others are."""
    units: NDA
    ids: LT
    paddings: BT  # If a position is a padding, we mark it as False. Otherwise True.
    lengths: LT = field(init=False)

    def __post_init__(self):
        self.ids.rename_('pos', 'batch')
        self.paddings.rename_('pos', 'batch')
        self.lengths = self.paddings.sum(dim='pos').rename_('batch')

    def __len__(self):
        return self.ids.size('batch')

    @property
    def num_units(self) -> int:
        return self.paddings.sum()


@batch_class
class OnePairBatch(BaseBatch):
    src_seqs: PaddedUnitSeqs
    tgt_seqs: PaddedUnitSeqs
    indices: LT  # This records the original indices in the dataset, i.e., in what order these tokens appear.

    def __post_init__(self):
        self.indices.rename_('batch')
        assert len(self.src_seqs) == len(self.tgt_seqs)

    def __len__(self):
        return len(self.src_seqs)

    @property
    def num_tgt_units(self) -> int:
        return self.tgt_seqs.num_units

    def cuda(self):
        super().cuda()
        self.src_seqs.cuda()
        self.tgt_seqs.cuda()
        return self


def _gather_from_batches(batches: List[Dict], item_name: str, is_seq: bool = True, is_tensor: bool = True):
    orig_lst = [batch[item_name] for batch in batches]

    if not is_tensor:
        return get_array(orig_lst)

    if not is_seq:
        ids = torch.from_numpy(np.asarray(orig_lst))
        return ids

    ids, paddings = pad_to_dense(orig_lst, dtype='l')
    ids = torch.from_numpy(ids.T)
    paddings = torch.from_numpy(paddings.T)
    return ids, paddings


def one_pair_collate_fn(batches: List[Dict]) -> OnePairBatch:

    src_ids, src_paddings = _gather_from_batches(batches, 'src_id_seq')
    tgt_ids, tgt_paddings = _gather_from_batches(batches, 'tgt_id_seq')
    src_units = _gather_from_batches(batches, 'src_unit_seq', is_tensor=False)
    tgt_units = _gather_from_batches(batches, 'tgt_unit_seq', is_tensor=False)
    indices = _gather_from_batches(batches, 'index', is_seq=False)

    src_seqs = PaddedUnitSeqs(src_units, src_ids, src_paddings)
    tgt_seqs = PaddedUnitSeqs(tgt_units, tgt_ids, tgt_paddings)

    return OnePairBatch(src_seqs, tgt_seqs, indices)


class OnePairDataLoader(BaseDataLoader):

    add_argument('batch_size', default=32, dtype=int, msg='Batch size.')

    collate_fn = one_pair_collate_fn

    def __init__(self,
                 task: Task,
                 split: Split,
                 data_path: Path,
                 src_lang: str,
                 tgt_lang: str,
                 src_abc: Alphabet,
                 tgt_abc: Alphabet):
        dataset = OnePairDataset(data_path, split, src_lang, tgt_lang, src_abc, tgt_abc)
        super().__init__(dataset, task, batch_size=g.batch_size)

    # IDEA(j_luo) Move this to core?
    def __iter__(self) -> Iterator[OnePairBatch]:
        for batch in super().__iter__():
            if has_gpus():
                yield batch.cuda()
            else:
                yield batch

    @cached_property
    def tgt_seqs(self) -> PaddedUnitSeqs:
        items = list()
        for i in range(len(self.dataset)):
            items.append(self.dataset[i])
        ids, paddings = (_gather_from_batches(items, 'tgt_id_seq'))
        units = _gather_from_batches(items, 'tgt_unit_seq', is_tensor=False)
        ret = PaddedUnitSeqs(units, ids, paddings)
        if has_gpus():
            ret.cuda()
        return ret

    def get_token_from_index(self, index: int, side: str):
        assert side in ['src', 'tgt']

        vocab = self.dataset.src_vocab if side == 'src' else self.dataset.tgt_vocab
        return vocab[index]


class DataLoaderRegistry(BaseDataLoaderRegistry):

    add_argument('data_path', dtype='path', msg='Path to the dataset.')
    add_argument('src_lang', dtype=str, msg='ISO code for the source language.')
    add_argument('tgt_lang', dtype=str, msg='ISO code for the target language.')

    def get_data_loader(self, task: Task, split: Split, src_abc: Alphabet, tgt_abc: Alphabet, **kwargs) -> BaseDataLoader:
        if task.name == 'one_pair':
            dl = OnePairDataLoader(task, split, g.data_path, g.src_lang, g.tgt_lang, src_abc, tgt_abc)
        else:
            raise ValueError(f'Cannot understand this task.')

        return dl

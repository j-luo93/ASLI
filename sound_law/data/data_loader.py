from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, SequentialSampler
from torch.utils.data.sampler import WeightedRandomSampler

import sound_law.rl.trajectory as tr
from sound_law.rl.action import SoundChangeActionSpace
from dev_misc import BT, LT, NDA, add_argument, g
from dev_misc.devlib import BaseBatch, batch_class, pad_to_dense
from dev_misc.devlib.helper import get_array, get_tensor, has_gpus
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import BaseSetting
from dev_misc.trainlib.base_data_loader import (BaseDataLoader,
                                                BaseDataLoaderRegistry)
from dev_misc.utils import cached_property, handle_sequence_inputs
from sound_law.data.alphabet import PAD_ID
from sound_law.data.cognate import CognateRegistry
from sound_law.data.dataset import OnePairDataset, Vocabulary, pad
from sound_law.data.setting import Setting

from .alphabet import Alphabet
from .cognate import CognateRegistry, postprocess
from .setting import Setting, Split


@batch_class
class PaddedUnitSeqs(BaseBatch):
    """`units` should not be transposed, but the others are."""
    lang: str
    forms: NDA
    units: NDA
    ids: LT
    paddings: BT  # If a position is a padding, we mark it as False. Otherwise True.
    lengths: LT = field(init=False)
    lang_id: Optional[int] = None  # This is only needed for one-to-many scenarios.

    def __post_init__(self):
        self.ids.rename_('pos', 'batch')
        self.paddings.rename_('pos', 'batch')
        self.lengths = self.paddings.sum(dim='pos').rename_('batch')
        assert self.ids.shape == self.paddings.shape

    def __len__(self):
        return self.ids.size('batch')

    @property
    def num_units(self) -> int:
        return self.paddings.sum()

    def split(self, size: int) -> List[PaddedUnitSeqs]:
        with NoName(self.ids, self.paddings):
            ids_lst = self.ids.split(size, dim=-1)
            paddings_lst = self.paddings.split(size, dim=-1)
        start = 0
        ret = list()
        for ids, paddings in zip(ids_lst, paddings_lst):
            length = ids.size(1)
            units = self.units[start: start + length]
            forms = self.forms[start: start + length]
            split = PaddedUnitSeqs(self.lang, forms, units, ids, paddings,
                                   lang_id=self.lang_id)
            ret.append(split)
            start += length
        assert start == self.ids.size('batch')
        return ret


@batch_class
class SourceOnlyBatch(BaseBatch):
    """This class only has source sequences."""
    src_seqs: PaddedUnitSeqs
    tgt_lang_id: int

    def __len__(self):
        return len(self.src_seqs)

    @classmethod
    def from_ipa_tokens(cls, ipa_tokens: str, abc: Alphabet, tgt_lang_id: int, sot: bool, eot: bool) -> SourceOnlyBatch:
        """Prepare a `SourceOnlyBatch object from just ipa tokens."""
        std_func = handle_sequence_inputs(lambda s: abc.standardize(s))
        record = postprocess(ipa_tokens.split(), std_func, abc)
        record['id_seq'] = pad(record['id_seq'], sot, eot, False)
        record['post_unit_seq'] = pad(record['post_unit_seq'], sot, eot, True)

        batches = [record]
        ids, paddings = _gather_from_batches(batches, 'id_seq')
        units = _gather_from_batches(batches, 'post_unit_seq', is_tensor=False)
        forms = _gather_from_batches(batches, 'form', is_seq=False, is_tensor=False)

        seqs = PaddedUnitSeqs(abc.lang, forms, units, ids, paddings)
        return cls(seqs, tgt_lang_id)

    def cuda(self):
        super().cuda()
        self.src_seqs.cuda()
        return self


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

    @property
    def tgt_lang_id(self) -> int:
        return self.tgt_seqs.lang_id


def _gather_from_batches(batches: List[Dict], item_name: str, is_seq: bool = True, is_tensor: bool = True):
    orig_lst = [batch[item_name] for batch in batches]

    if not is_tensor:
        return get_array(orig_lst)

    if not is_seq:
        ids = torch.from_numpy(np.asarray(orig_lst))
        return ids

    ids, paddings = pad_to_dense(orig_lst, dtype='l', pad_idx=PAD_ID)
    ids = torch.from_numpy(ids.T)
    paddings = torch.from_numpy(paddings.T)
    return ids, paddings


def one_pair_collate_fn(batches: List[Dict]) -> OnePairBatch:

    src_ids, src_paddings = _gather_from_batches(batches, 'src_id_seq')
    tgt_ids, tgt_paddings = _gather_from_batches(batches, 'tgt_id_seq')
    src_units = _gather_from_batches(batches, 'src_unit_seq', is_tensor=False)
    tgt_units = _gather_from_batches(batches, 'tgt_unit_seq', is_tensor=False)
    src_forms = _gather_from_batches(batches, 'src_form', is_seq=False, is_tensor=False)
    tgt_forms = _gather_from_batches(batches, 'tgt_form', is_seq=False, is_tensor=False)
    indices = _gather_from_batches(batches, 'index', is_seq=False)

    src_lang = batches[0]['src_lang']
    tgt_lang = batches[0]['tgt_lang']
    src_seqs = PaddedUnitSeqs(src_lang, src_forms, src_units, src_ids, src_paddings)
    tgt_seqs = PaddedUnitSeqs(tgt_lang, tgt_forms, tgt_units, tgt_ids, tgt_paddings)

    return OnePairBatch(src_seqs, tgt_seqs, indices)


class BaseOnePairDataLoader(BaseDataLoader):

    collate_fn = one_pair_collate_fn
    dataset: OnePairDataset

    def _base_init(self,
                   setting: Setting,
                   cog_reg: CognateRegistry,
                   lang2id: Dict[str, int] = None):
        """Perform initialization for base class. and return dataset."""
        dataset = cog_reg.prepare_dataset(setting)
        self.lang2id = lang2id
        self.src_lang = setting.src_lang
        self.tgt_lang = setting.tgt_lang
        self.src_abc = cog_reg.get_alphabet(setting.src_lang)
        self.tgt_abc = cog_reg.get_alphabet(setting.tgt_lang)
        return dataset

    def _postprocess_batch(self, batch: OnePairBatch) -> OnePairBatch:
        if self.lang2id is not None:
            # NOTE(j_luo) Source lang id not needed for now.
            batch.tgt_seqs.lang_id = self.lang2id[batch.tgt_seqs.lang]
        if has_gpus():
            return batch.cuda()
        return batch


class OnePairDataLoader(BaseOnePairDataLoader):

    def __init__(self,
                 setting: Setting,
                 cog_reg: CognateRegistry,
                 lang2id: Dict[str, int] = None):
        dataset = self._base_init(setting, cog_reg, lang2id)

        sampler = None
        if setting.for_training:
            sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset))
        super().__init__(dataset, setting,
                         batch_size=g.batch_size,
                         sampler=sampler)

    def __iter__(self) -> Iterator[OnePairBatch]:
        for batch in super().__iter__():
            yield self._postprocess_batch(batch)

    @cached_property
    def tgt_seqs(self) -> PaddedUnitSeqs:
        vocab = self.tgt_vocabulary
        items = list()
        for i in range(len(vocab)):
            items.append(vocab[i])
        ids, paddings = (_gather_from_batches(items, 'id_seq'))
        units = _gather_from_batches(items, 'unit_seq', is_tensor=False)
        forms = _gather_from_batches(items, 'form', is_tensor=False, is_seq=False)
        ret = PaddedUnitSeqs(self.tgt_lang, forms, units, ids, paddings)
        if has_gpus():
            ret.cuda()
        return ret

    @property
    def src_vocabulary(self) -> Vocabulary:
        return self.dataset.src_vocabulary

    @property
    def tgt_vocabulary(self) -> Vocabulary:
        return self.dataset.tgt_vocabulary


class VSOnePairDataLoader(BaseOnePairDataLoader):  # VS stands for vocab state.
    # FIXME(j_luo) Need to handle duplicates.
    """This data loader always return the entire dataset as one fixed batch."""

    def __init__(self,
                 setting: Setting,
                 cog_reg: CognateRegistry,
                 lang2id: Dict[str, int] = None):
        if setting.src_sot != setting.tgt_sot:
            raise ValueError(f'Expect equal values, but got {setting.src_sot} and {setting.tgt_sot}.')
        if setting.src_eot != setting.tgt_eot:
            raise ValueError(f'Expect equal values, but got {setting.src_eot} and {setting.tgt_eot}.')

        dataset = self._base_init(setting, cog_reg, lang2id)
        ds = len(dataset)  # Data size.
        # A batch sampler that samples the entire dataset in a fixed order.
        batch_sampler = BatchSampler(SequentialSampler(range(ds)), batch_size=ds, drop_last=False)
        super().__init__(dataset, setting,
                         batch_sampler=batch_sampler)
        # This is used to cache the entire batch.
        self._entire_batch: OnePairBatch = None

    @property
    def entire_batch(self) -> OnePairBatch:
        # Obtain the entire batch for the first time only.
        if self._entire_batch is None:
            lst = list(super().__iter__())
            if len(lst) != 1:
                raise RuntimeError(f"Expecting exactly one batch but got {len(lst)} instead.")
            self._entire_batch = lst[0]
            if has_gpus():
                self._entire_batch = self._entire_batch.cuda()
            # Rename `batch` to `word`.
            self._entire_batch.src_seqs.ids.rename_(batch='word')
            self._entire_batch.tgt_seqs.ids.rename_(batch='word')

        return self._entire_batch

    def __iter__(self) -> Iterator[OnePairBatch]:
        yield self._postprocess_batch(self.entire_batch)


class DataLoaderRegistry(BaseDataLoaderRegistry):

    add_argument('data_path', dtype='path', msg='Path to the dataset.')
    add_argument('src_lang', dtype=str, msg='ISO code for the source language.')
    add_argument('tgt_lang', dtype=str, msg='ISO code for the target language.')
    add_argument('input_format', dtype=str, choices=['wikt', 'ielex'], default='ielex', msg='Input format.')

    def get_data_loader(self, setting: BaseSetting, cog_reg: CognateRegistry, **kwargs) -> BaseDataLoader:
        if setting.task == 'one_pair':
            # TODO(j_luo) The options can all be part of setting.
            dl_cls = VSOnePairDataLoader if g.use_rl else OnePairDataLoader
            dl = dl_cls(setting, cog_reg, **kwargs)
        else:
            raise ValueError(f'Cannot understand this task "{setting.task}".')
        return dl

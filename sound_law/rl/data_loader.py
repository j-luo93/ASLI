from __future__ import annotations

from typing import Dict

from torch.utils.data import BatchSampler, SequentialSampler

from dev_misc.trainlib.base_data_loader import BaseDataLoader
from sound_law.data.cognate import CognateRegistry
from sound_law.data.data_loader import (OnePairBatch, OnePairDataLoader,
                                        PaddedUnitSeqs)
from sound_law.data.setting import Setting


class EntireBatchOnePairDataLoader(OnePairDataLoader):
    # FIXME(j_luo) This should be merged with OnePairDataLoader or sth like that.
    """This data loader always return the entire dataset as one fixed batch."""

    def __init__(self,
                 setting: Setting,
                 cog_reg: CognateRegistry,
                 lang2id: Dict[str, int] = None):
        dataset = cog_reg.prepare_dataset(setting)
        self.lang2id = lang2id
        self.src_lang = setting.src_lang
        self.tgt_lang = setting.tgt_lang
        self.src_abc = cog_reg.get_alphabet(setting.src_lang)
        self.tgt_abc = cog_reg.get_alphabet(setting.tgt_lang)

        ds = len(dataset)  # Data size.
        # A batch sampler that samples the entire dataset in a fixed order.
        batch_sampler = BatchSampler(SequentialSampler(range(ds)), batch_size=ds, drop_last=False)
        BaseDataLoader.__init__(self, dataset, setting,
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

        return self._entire_batch

    def __iter__(self):
        yield self.entire_batch

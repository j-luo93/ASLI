from pathlib import Path
from typing import Dict, Iterator, List

import torch

from dev_misc import LT, add_argument, g
from dev_misc.devlib import BaseBatch, batch_class, pad_to_dense
from dev_misc.trainlib import Task
from dev_misc.trainlib.base_data_loader import (BaseDataLoader,
                                                BaseDataLoaderRegistry)
from sound_law.data.dataset import OnePairDataset


@batch_class
class OnePairBatch(BaseBatch):
    src_id_seqs: LT
    tgt_id_seqs: LT

    def __len__(self):
        return self.src_id_seqs.size('batch')


def one_pair_collate_fn(batches: List[Dict]) -> OnePairBatch:
    src_id_seqs = pad_to_dense([batch['src_id_seq'] for batch in batches])
    tgt_id_seqs = pad_to_dense([batch['tgt_id_seq'] for batch in batches])

    src_id_seqs = torch.from_numpy(src_id_seqs.T)
    tgt_id_seqs = torch.from_numpy(tgt_id_seqs.T)

    return OnePairBatch(src_id_seqs, tgt_id_seqs)


class OnePairDataLoader(BaseDataLoader):

    collate_fn = one_pair_collate_fn

    def __init__(self, task: Task, data_path: Path, src_lang: str, tgt_lang: str):
        dataset = OnePairDataset(data_path, src_lang, tgt_lang)
        super().__init__(dataset, task)

    # IDEA(j_luo) Move this to core?
    def __iter__(self) -> Iterator[OnePairBatch]:
        for batch in super().__iter__():
            yield batch.cuda()


class DataLoaderRegistry(BaseDataLoaderRegistry):

    add_argument('data_path', dtype='path', msg='Path to the dataset.')
    add_argument('src_lang', dtype=str, msg='ISO code for the source language.')
    add_argument('tgt_lang', dtype=str, msg='ISO code for the target language.')

    def get_data_loader(self, task: Task, *args, **kwargs) -> BaseDataLoader:
        if task.name == 'one_pair':
            dl = OnePairDataLoader(task, g.data_path, g.src_lang, g.tgt_lang)
        else:
            raise ValueError(f'Cannot understand this task.')

        return dl

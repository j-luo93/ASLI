from pathlib import Path
from typing import List, overload

import pandas as pd
from torch.utils.data import Dataset


class Alphabet:
    """A class to represent the alphabet of any dataset."""

    def __init__(self, contents: List[List[str]]):
        data = set()
        for content in contents:
            data.update(content)
        self._id2unit = sorted(data)
        self._unit2id = {c: i for i, c in enumerate(self._id2unit)}

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


class OnePairDataset(Dataset):

    def __init__(self, data_path: Path, src_lang: str, tgt_lang: str):
        src_path = data_path / f'{src_lang}.tsv'
        tgt_path = data_path / f'{tgt_lang}.tsv'

        src_df = pd.read_csv(str(src_path), sep='\t')
        tgt_df = pd.read_csv(str(tgt_path), sep='\t')

        self.src_unit_seqs = src_df['tokens'].str.split().to_list()
        self.tgt_unit_seqs = tgt_df['tokens'].str.split().to_list()

        self.src_abc = Alphabet(self.src_unit_seqs)
        self.tgt_abc = Alphabet(self.tgt_unit_seqs)

        self.src_id_seqs = [[self.src_abc[u] for u in seq] for seq in self.src_unit_seqs]
        self.tgt_id_seqs = [[self.tgt_abc[u] for u in seq] for seq in self.tgt_unit_seqs]

    def __getitem__(self, index: int):
        return {
            'src_id_seq': self.src_id_seqs[index],
            'tgt_id_seq': self.tgt_id_seqs[index]
        }

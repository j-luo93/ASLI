import logging
from pathlib import Path
from typing import List, overload

import pandas as pd
from torch.utils.data import Dataset

from dev_misc.devlib.helper import get_array

SOT = '<SOT>'
EOT = '<EOT>'
SOT_ID = 0
EOT_ID = 1


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

        logging.info(f'Alphabet for {lang}: {self._id2unit}.')

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


class OnePairDataset(Dataset):

    def __init__(self, data_path: Path, src_lang: str, tgt_lang: str):
        src_path = data_path / f'{src_lang}.tsv'
        tgt_path = data_path / f'{tgt_lang}.tsv'

        src_df = pd.read_csv(str(src_path), sep='\t')
        tgt_df = pd.read_csv(str(tgt_path), sep='\t')

        self.src_vocab = get_array(src_df['transcription'])
        self.tgt_vocab = get_array(tgt_df['transcription'])

        self.src_unit_seqs = get_array(src_df['tokens'].str.split().to_list())
        self.tgt_unit_seqs = get_array(tgt_df['tokens'].str.split().to_list())

        self.src_abc = Alphabet(src_lang, self.src_unit_seqs)
        self.tgt_abc = Alphabet(tgt_lang, self.tgt_unit_seqs)

        self.src_id_seqs = [[self.src_abc[u] for u in seq] for seq in self.src_unit_seqs]
        self.tgt_id_seqs = [[self.tgt_abc[u] for u in seq] for seq in self.tgt_unit_seqs]

    def __getitem__(self, index: int):
        return {
            'src_id_seq': self.src_id_seqs[index],
            'src_unit_seq': self.src_unit_seqs[index],
            'tgt_id_seq': self.tgt_id_seqs[index] + [EOT_ID],
            'tgt_unit_seq': self.tgt_unit_seqs[index] + [EOT],
            'index': index
        }

    def __len__(self):
        return len(self.src_id_seqs)

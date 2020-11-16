from dev_misc import NDA
from typing import Optional
import logging
from collections import Counter, defaultdict
from typing import List, Union, overload

import numpy as np
import pandas as pd
import torch
from panphon.featuretable import FeatureTable

from dev_misc import LT, g

SOT = '<SOT>'
EOT = '<EOT>'
PAD = '<pad>'
SOT_ID = 0
EOT_ID = 1
PAD_ID = 2

_ft = FeatureTable()


class Alphabet:
    """A class to represent the alphabet of any dataset."""

    def __init__(self, lang: str, contents: List[List[str]], sources: Union[str, List[str]], dist_mat: Optional[NDA] = None):
        if sources is not None:
            if isinstance(sources, str):
                sources = [sources] * len(contents)
            else:
                assert len(contents) == len(sources)
        else:
            sources = ['unknown'] * len(contents)

        cnt = defaultdict(Counter)
        for content, source in zip(contents, sources):
            for u in content:
                cnt[u][source] += 1

        # Merge symbols with identical phonological features if needed.
        if not g.use_mcts and not g.use_duplicate_phono and g.use_phono_features:
            t2u = defaultdict(list)  # tuple-to-units
            for u in cnt:
                t = tuple(self.get_pfv(u).numpy())
                t2u[t].append(u)

            u2u = dict()  # unit-to-unit. This finds the standardized unit.
            for units in t2u.values():
                lengths = [len(u) for u in units]
                min_i = lengths.index(min(lengths))
                std_u = units[min_i]
                for u in units:
                    u2u[u] = std_u

            merged_cnt = defaultdict(Counter)
            for u, std_u in u2u.items():
                merged_cnt[std_u].update(cnt[u])

            logging.imp(f'Symbols are merged based on phonological features: from {len(cnt)} to {len(merged_cnt)}.')
            cnt = merged_cnt
            self._u2u = u2u

        units = sorted(cnt.keys())
        self.special_units = [SOT, EOT, PAD]
        self.special_ids = [SOT_ID, EOT_ID, PAD_ID]
        self._id2unit = self.special_units + units
        self._unit2id = dict(zip(self.special_units, self.special_ids))
        self._unit2id.update({c: i for i, c in enumerate(units, len(self.special_units))})
        self.stats: pd.DataFrame = pd.DataFrame.from_dict(cnt)
        self.dist_mat = None
        if dist_mat is not None:
            # Pad the dist_mat for special units.
            self.dist_mat = np.full([len(self), len(self)], 99999, dtype='long')
            orig_ids = np.asarray([self[u] for u in contents[0]])
            self.dist_mat[orig_ids.reshape(-1, 1), orig_ids] = dist_mat

        logging.info(f'Alphabet for {lang}, size {len(self._id2unit)}: {self._id2unit}.')
        self.lang = lang

    @property
    def pfm(self) -> LT:
        """Phonological feature matrix for the entire alphabet. For the special units, use all 0's."""
        pfvs = [torch.zeros(22).long() for _ in range(len(self.special_ids))]
        for unit in self._id2unit[len(self.special_ids):]:
            pfv = self.get_pfv(unit)
            pfvs.append(pfv)
        pfm = torch.stack(pfvs, dim=0).refine_names(..., 'phono_feat')
        return pfm

    def standardize(self, s: str) -> str:
        """Standardize the string if needed."""
        if not g.use_mcts and not g.use_duplicate_phono and g.use_phono_features:
            return self._u2u[s]
        return s

    def get_pfv(self, s: str) -> LT:
        """Get phonological feature vector (pfv) for a unit."""
        ret = _ft.word_to_vector_list(s, numeric=True)
        if len(ret) != 1:
            raise ValueError(f'Inconsistent tokenization results between panphon and lingpy.')

        # NOTE(j_luo) `+1` since the original features range from -1 to 1.
        ret = torch.LongTensor(ret[0]) + 1
        return ret

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, unit: str) -> int: ...

    def __getitem__(self, index_or_unit):
        if isinstance(index_or_unit, int) or np.issubdtype(type(index_or_unit), np.integer):
            return self._id2unit[index_or_unit]
        elif isinstance(index_or_unit, str):
            return self._unit2id[index_or_unit]
        else:
            raise TypeError(f'Unsupported type for "{index_or_unit}".')

    def __len__(self):
        return len(self._unit2id)

    def __iter__(self):
        yield from self._id2unit

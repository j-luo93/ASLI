from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from dev_misc.trainlib import BaseSetting

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


@dataclass
class Setting(BaseSetting):
    task: str
    split: Split
    src_lang: str
    tgt_lang: str
    # This indicates whether a data loader is used for training. If `True`, weighted random sampling is used for batches.
    for_training: bool
    keep_ratio: Optional[float] = None
    # Flags to control how src/tgt tokens are padded.
    # By default, SOT and EOT are added on both sides except for
    # tgt, SOT is ignored.
    src_sot: bool = True
    src_eot: bool = True
    tgt_sot: bool = False
    tgt_eot: bool = True

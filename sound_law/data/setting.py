from dataclasses import dataclass
from typing import Optional

from dev_misc.trainlib import BaseSetting

from .dataset import Split


@dataclass
class Setting(BaseSetting):
    task: str
    split: Split
    src_lang: str
    tgt_lang: str
    # This indicates whether a data loader is used for training. If `True`, weighted random sampling is used for batches.
    for_training: bool
    keep_ratio: Optional[float] = None

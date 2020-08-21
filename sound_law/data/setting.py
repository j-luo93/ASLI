from dataclasses import dataclass

from dev_misc.trainlib import BaseSetting

from .dataset import Alphabet, Split


@dataclass
class Setting(BaseSetting):
    task: str
    split: Split
    src_lang: str
    tgt_lang: str
    src_abc: Alphabet
    tgt_abc: Alphabet
    # This indicates whether a data loader is used for training. If `True`, weighted random sampling is used for batches.
    for_training: bool

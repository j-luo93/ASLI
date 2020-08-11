from pathlib import Path
from typing import Tuple

from dev_misc.arglib import Registry

reg = Registry('config')


@reg
class ZSLatIta:  # "ZS" stands for zero-shot.
    data_path: Path = Path('data/wikt')
    src_lang: str = 'lat'
    tgt_lang: str = 'ita'
    dropout: float = 0.2
    check_interval: int = 100
    num_steps: int = 10000
    input_format: str = 'wikt'
    eval_interval: int = 1000
    control_mode: str = 'none'
    train_tgt_langs: Tuple[str] = ('ron', 'cat', 'spa', 'por')
    task: str = 'one_to_many'
    lang_emb_mode: str = 'mean'


@reg
class ZSLatSpa(ZSLatIta):
    tgt_lang: str = 'spa'
    train_tgt_langs: Tuple[str] = ('ron', 'cat', 'ita', 'por')


@reg
class ZSPgmcDeu(ZSLatIta):
    src_lang: str = 'pgmc'
    tgt_lang: str = 'deu'
    train_tgt_langs: Tuple[str] = ('swe', 'nld')

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dev_misc.arglib import Registry

reg = Registry('config')


@reg
class ZSLatIta:  # "ZS" stands for zero-shot.
    share_src_tgt_abc: bool = True
    data_path: Path = Path('data/wikt')
    src_lang: str = 'lat'
    tgt_lang: str = 'ita'
    dropout: float = 0.2
    check_interval: int = 100
    num_steps: int = 10000
    input_format: str = 'wikt'
    eval_interval: int = 1000
    control_mode: str = 'none'
    train_tgt_langs: Tuple[str, ...] = ('ron', 'cat', 'spa', 'por')
    task: str = 'one_to_many'
    lang_emb_mode: str = 'mean'
    beam_size: int = 5


@reg
class ZSLatSpa(ZSLatIta):
    tgt_lang: str = 'spa'
    train_tgt_langs: Tuple[str, ...] = ('ron', 'cat', 'ita', 'por')


@reg
class OPLatSpa(ZSLatSpa):  # "OP" stands for one-pair.
    task: str = 'one_pair'


@reg
class ZSPgmcDeu(ZSLatIta):
    src_lang: str = 'pgmc'
    tgt_lang: str = 'deu'
    train_tgt_langs: Tuple[str, ...] = ('swe', 'nld')


@reg
class ZSPgmcNld(ZSPgmcDeu):
    tgt_lang: str = 'nld'
    train_tgt_langs: Tuple[str, ...] = ('swe', 'deu')


@dataclass
class Size220:
    # the char_emb_size and hidden_size must be multiples of 22 since there are 22 phonological features being used (each of which has its own embedding)
    char_emb_size: int = 220
    hidden_size: int = 220


@dataclass
class Size440:
    char_emb_size: int = 440
    hidden_size: int = 440


@dataclass
class Size880:
    char_emb_size: int = 880
    hidden_size: int = 880


@dataclass
class UsePhono:
    use_phono_features: bool = True
    share_src_tgt_abc: bool = True
    use_duplicate_phono: bool = False


@reg
class ZSLatItaPhono(ZSLatIta, UsePhono, Size220):
    pass


@reg
class ZSLatSpaPhono(ZSLatSpa, UsePhono, Size220):
    pass


@reg
class ZSPgmcDeuPhono(ZSPgmcDeu, UsePhono, Size220):
    pass


@reg
class ZSPgmcNldPhono(ZSPgmcNld, UsePhono, Size220):
    pass


@reg
class OPLatSpaPhono(ZSLatSpaPhono, Size220):  # "OP" stands for one-pair.
    task: str = 'one_pair'


@reg
class OPLatSpaPhono880(Size880, OPLatSpaPhono):
    pass


@reg
class CnnZSLatIta(ZSLatIta):
    model_encoder_type: str = 'cnn'
    kernel_sizes: Tuple[int, ...] = (3, 5, 7)


@reg
class CnnZSLatItaPhono(CnnZSLatIta, UsePhono, Size220):
    pass

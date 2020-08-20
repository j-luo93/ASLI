from dataclasses import dataclass
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


@dataclass
class UsePhono:
    use_phono_features: bool = True
    share_src_tgt_abc: bool = True
    char_emb_size: int = 220
    hidden_size: int = 220


@reg
class ZSLatItaPhono(ZSLatIta, UsePhono):
    pass


@reg
class ZSLatSpaPhono(ZSLatSpa, UsePhono):
    pass

@reg
class CnnZSLatIta(ZSLatIta):
    model_encoder_type: str = 'cnn'
    kernel_sizes: Tuple[int, ...] = (3, 5, 7)

@reg
class CnnZSLatItaPhono(CnnZSLatIta, UsePhono):
    pass


# I will be trying out the following things in this grid search:
# (without phono features)
# char_emb_size/hidden_size: 128, 256, 512
# (with phono features)
# char_emb_size/hidden_size: 110, 220, 440
# (for both)
# lstm/cnn encoder
# if cnn encoder, kernel sizes — and eventually different Cnn Encoder architectures
# dropout 0, 0.2, 0.4, 0.6
# different src/tgt lang pairs within wikt data

from dataclasses import dataclass, make_dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

from inflection import camelize

from dev_misc.arglib import Registry

reg = Registry('config')
a2c_reg = Registry('a2c_config')
mcts_reg = Registry('mcts_config')
s2s_reg = Registry('s2s_reg')


@s2s_reg
class S2S:
    check_interval: int = 100


@reg
class ZSLatIta:  # "ZS" stands for zero-shot.
    share_src_tgt_abc: bool = True
    data_path: Path = Path('data/wikt')
    src_lang: str = 'lat'
    tgt_lang: str = 'ita'
    dropout: float = 0.2
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
    train_tgt_langs: Tuple[str, ...] = None


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


# Programmatically create several configs.

def iter_tgt_lang(all_langs: List[str]) -> Iterator[Tuple[str, Tuple[str, ...]]]:
    """An iterator that returns a tuple of `(tgt_lang, train_tgt_langs)`."""
    for tgt_lang in all_langs:
        train_tgt_langs = tuple(lang for lang in all_langs if lang != tgt_lang)
        yield tgt_lang, train_tgt_langs


def register_phono_nel_configs(all_langs: List[str], proto: str, proto_code: str) -> list:
    """Register phonological configs with NorthEuraLex dataset.

    Note that `proto` is used for the config name, and `proto_code` is the actual language code
    for identifying this language in the dataset.
    """
    configs = list()
    for tgt_lang, train_tgt_langs in iter_tgt_lang(all_langs):
        cls_name = f'ZS{camelize(proto)}{camelize(tgt_lang)}PhonoNel'
        new_cls = make_dataclass(cls_name,
                                 [('src_lang', str, proto_code),
                                  ('tgt_lang', str, tgt_lang),
                                  ('train_tgt_langs', Tuple[str, ...], train_tgt_langs)],
                                 bases=(ZSPgmcDeuPhono, ))
        configs.append(reg(new_cls))
    return configs


all_germanic_langs = ['deu', 'swe', 'nld', 'isl', 'nor', 'dan', 'eng']
all_italic_langs = ['por', 'spa', 'ita', 'fra', 'ron', 'cat']

all_germanic_configs = register_phono_nel_configs(all_germanic_langs, 'pgmc', 'gem-pro')
all_italic_configs = register_phono_nel_configs(all_italic_langs, 'lat', 'la')


@reg
class ZSPgmcNldPhono(ZSPgmcNld, UsePhono, Size220):
    pass


@reg
class OPLatSpaPhono(ZSLatSpaPhono, Size220):  # "OP" stands for one-pair.
    task: str = 'one_pair'


@a2c_reg
class Ppo:
    critic_target: str = 'expected'
    learning_rate: float = 5e-4
    value_learning_rate: float = 2e-3
    value_steps: int = 50
    check_interval: int = 10
    batch_size: int = 512
    use_gae: bool = True
    gae_lambda: float = 0.2
    init_entropy_reg: float = 0.01
    end_entropy_reg: float = 1e-4
    when_entropy_reg: int = 100
    policy_steps: int = 50
    use_ppo: bool = True


@reg
class OPRLFake(OPLatSpaPhono):
    use_rl: bool = True
    agent: str = 'a2c'
    num_layers: int = 1
    use_phono_features: bool = False
    factorize_actions: bool = True

    src_lang: str = 'fake1'
    tgt_lang: str = 'fake2'


@reg
class OPRLFake50(OPRLFake):
    src_lang: str = 'fake1-50'
    tgt_lang: str = 'fake2-50'


@reg
class OPRLFakeR2(OPRLFake):
    src_lang: str = 'fake1-r2'
    tgt_lang: str = 'fake2-r2'


@reg
class OPRLFakeR3(OPRLFake):
    src_lang: str = 'fake1-r3'
    tgt_lang: str = 'fake2-r3'


@reg
class OPRLFakeR4(OPRLFake):
    src_lang: str = 'fake1-r4'
    tgt_lang: str = 'fake2-r4'


@reg
class OPRLFakeR5(OPRLFake):
    when_entropy_reg: int = 200
    src_lang: str = 'fake1-r5'
    tgt_lang: str = 'fake2-r5'


@reg
class OPRLFakeR5C(OPRLFakeR5):
    src_lang: str = 'fake1-r5c'
    tgt_lang: str = 'fake2-r5c'


@reg
class OPRLFakeR10(OPRLFake):
    max_rollout_length: int = 15
    src_lang: str = 'fake1-r10'
    tgt_lang: str = 'fake2-r10'


@reg
class OPRLFakeR15(OPRLFake):
    max_rollout_length: int = 20
    src_lang: str = 'fake1-r15'
    tgt_lang: str = 'fake2-r15'


@reg
class OPRLFakeR20(OPRLFake):
    max_rollout_length: int = 25
    src_lang: str = 'fake1-r20'
    tgt_lang: str = 'fake2-r20'


@reg
class OPRLFakeR30(OPRLFake):
    max_rollout_length: int = 35
    src_lang: str = 'fake1-r30'
    tgt_lang: str = 'fake2-r30'


@reg
class OPRLFakeR30C(OPRLFakeR30):
    src_lang: str = 'fake1-r30c'
    tgt_lang: str = 'fake2-r30c'


@reg
class OPRLFakeR5i(OPRLFake):
    src_lang: str = 'fake1-r5i'
    tgt_lang: str = 'fake2-r5i'


@mcts_reg
class BasicMcts:
    use_mcts: bool = True
    check_interval: int = 1
    expansion_batch_size: int = 10
    num_mcts_sims: int = 40
    num_inner_steps: int = 50
    learning_rate: float = 1e-3
    virtual_loss: float = 0.5
    game_count: int = 3
    num_workers: int = 10
    mixing: float = 0.0
    puct_c: float = 5.0
    num_episodes: int = 500
    episode_check_interval: int = 50
    dirichlet_alpha: float = 0.3
    phoible_path: 'path' = 'data/phoible_segs.pkl'
    use_finite_horizon: bool = True
    use_max_value: bool = True


@mcts_reg
class MoreSims(BasicMcts):
    expansion_batch_size: int = 80
    episode_check_interval: int = 10
    num_mcts_sims: int = 4000
    num_episodes: int = 10


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

"""
A manager class takes care of managing the data loader, the model, and the trainer.
"""
import logging

from torch.optim import Adam

from dev_misc import add_argument, g
from dev_misc.devlib.helper import has_gpus
from sound_law.data.data_loader import DataLoaderRegistry
from sound_law.data.dataset import Alphabet, Split, get_paths
from sound_law.data.setting import Setting
from sound_law.evaluate.evaluator import Evaluator
from sound_law.model.one_pair import OnePairModel
from sound_law.model.one_to_many import OneToManyModel

from .trainer import Trainer

add_argument('check_interval', default=10, dtype=int, msg='Frequency to check the training progress.')
add_argument('eval_interval', default=100, dtype=int, msg='Frequency to call the evaluator.')


class OnePairManager:
    """A manager for sound law induction on one src-tgt pair."""

    def __init__(self):
        # Prepare alphabets first.
        src_path, tgt_path = get_paths(g.data_path, g.src_lang, g.tgt_lang)
        self.src_abc = Alphabet.from_tsv(g.src_lang, src_path, g.input_format)
        self.tgt_abc = Alphabet.from_tsv(g.tgt_lang, tgt_path, g.input_format)

        # Prepare data loaders with different splits.
        self.dl_reg = DataLoaderRegistry()

        def create_setting(name: str, split: Split, for_training: bool) -> Setting:
            return Setting(name, 'one_pair', split, g.src_lang, g.tgt_lang, self.src_abc, self.tgt_abc, for_training)

        if g.input_format == 'wikt':
            train_setting = create_setting('train', Split('train'), True)
            train_e_setting = create_setting('train_e', Split('train'), False)  # For evaluation.
            dev_setting = create_setting('dev', Split('dev'), False)
            self.dl_reg.register_data_loader(train_setting)
            self.dl_reg.register_data_loader(train_e_setting)
            self.dl_reg.register_data_loader(dev_setting)
        else:
            for fold in range(5):
                dev_fold = fold + 1
                train_folds = list(range(1, 6))
                train_folds.remove(dev_fold)

                train_setting = create_setting(f'train@{fold}', Split('train', train_folds), True)
                train_e_setting = create_setting(f'train@{fold}_e', Split('train', train_folds), False)
                dev_setting = create_setting(f'dev@{fold}', Split('dev', [dev_fold]), False)
                self.dl_reg.register_data_loader(train_setting)
                self.dl_reg.register_data_loader(train_e_setting)
                self.dl_reg.register_data_loader(dev_setting)

        test_setting = create_setting('test', Split('test'), False)
        self.dl_reg.register_data_loader(test_setting)

    def run(self):

        def run_once(train_name, dev_name, test_name):
            train_dl = self.dl_reg[train_name]
            train_e_dl = self.dl_reg[f'{train_name}_e']
            dev_dl = self.dl_reg[dev_name]
            test_dl = self.dl_reg[test_name]

            model = OnePairModel(len(self.src_abc), len(self.tgt_abc))
            if has_gpus():
                model.cuda()
            evaluator = Evaluator(model, {train_name: train_e_dl, dev_name: dev_dl, test_name: test_dl})

            trainer = Trainer(model, [self.dl_reg.get_setting_by_name(train_name)],
                              [1.0], 'step',
                              check_interval=g.check_interval,
                              evaluator=evaluator,
                              eval_interval=g.eval_interval)
            trainer.init_params('uniform', -0.1, 0.1)
            trainer.set_optimizer(Adam, lr=0.002)
            trainer.train(self.dl_reg)

        if g.input_format == 'wikt':
            run_once('train', 'dev', 'test')
        else:
            for fold in range(5):
                logging.imp(f'Cross-validation, fold number {fold}')
                run_once(f'train@{fold}', f'dev@{fold}', 'test')


class OneToManyManager:
    """The manager class for single-source-multiple-target scenarios."""

    add_argument('train_tgt_langs', dtype=str, nargs='+', msg='Target languages used for training.')

    def __init__(self):
        # Get alphabets from tsvs. Note that the target alphabet is based on the union of all target languages, i.e., a shared alphabet for all.
        src_paths = list()
        tgt_paths = list()
        all_tgt = sorted([g.tgt_lang] + list(g.train_tgt_langs))
        for tgt in all_tgt:
            src_path, tgt_path = get_paths(g.data_path, g.src_lang, tgt)
            src_paths.append(src_path)
            tgt_paths.append(tgt_path)
        self.src_abc = Alphabet.from_tsvs(g.src_lang, src_paths, g.input_format)
        self.tgt_abc = Alphabet.from_tsvs('all_targets', tgt_paths, g.input_format)

        # Get language-to-id mappings. Used only for the targets (i.e., decoder side).
        lang2id = {tgt: i for i, tgt in enumerate(all_tgt)}

        # Get all data loaders.
        self.dl_reg = DataLoaderRegistry()

        def create_setting(name: str, tgt_lang: str, split: Split, for_training: bool) -> Setting:
            return Setting(name, 'one_pair', split, g.src_lang, tgt_lang, self.src_abc, self.tgt_abc, for_training)

        test_setting = create_setting(f'test@{g.tgt_lang}', g.tgt_lang, Split('all'), False)
        self.dl_reg.register_data_loader(test_setting, lang2id=lang2id)

        # Get the training languages.
        for train_tgt_lang in g.train_tgt_langs:
            if g.input_format == 'ielex':
                train_split = Split('train', [1, 2, 3, 4])  # Use the first four folds for training.
                dev_split = Split('dev', [5])  # Use the last fold for dev.
            else:
                train_split = Split('train')
                dev_split = Split('dev')
            train_setting = create_setting(f'train@{train_tgt_lang}', train_tgt_lang, train_split, True)
            train_e_setting = create_setting(f'train@{train_tgt_lang}_e', train_tgt_lang, train_split, False)
            dev_setting = create_setting(f'dev@{train_tgt_lang}', train_tgt_lang, dev_split, False)
            test_setting = create_setting(f'test@{train_tgt_lang}', train_tgt_lang, Split('test'), False)

            self.dl_reg.register_data_loader(train_setting, lang2id=lang2id)
            self.dl_reg.register_data_loader(train_e_setting, lang2id=lang2id)
            self.dl_reg.register_data_loader(dev_setting, lang2id=lang2id)
            self.dl_reg.register_data_loader(test_setting, lang2id=lang2id)

        self.model = OneToManyModel(len(self.src_abc), len(self.tgt_abc), len(g.train_tgt_langs) + 1)
        if has_gpus():
            self.model.cuda()

        # NOTE(j_luo) Evaluate on every loader that is not for training.
        eval_dls = self.dl_reg.get_loaders_by_name(lambda name: 'train' not in name or '_e' in name)
        self.evaluator = Evaluator(self.model, eval_dls)

        train_names = [f'train@{train_tgt_lang}' for train_tgt_lang in g.train_tgt_langs]
        train_settings = [self.dl_reg.get_setting_by_name(name) for name in train_names]
        self.trainer = Trainer(self.model, train_settings,
                               [1.0] * len(train_settings), 'step',
                               evaluator=self.evaluator,
                               check_interval=g.check_interval,
                               eval_interval=g.eval_interval)
        self.trainer.init_params('uniform', -0.1, 0.1)
        self.trainer.set_optimizer(Adam, lr=0.002)

    def run(self):
        self.trainer.train(self.dl_reg)

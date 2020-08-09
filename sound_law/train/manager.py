"""
A manager class takes care of managing the data loader, the model, and the trainer.
"""
from dataclasses import dataclass
import logging
from typing import ClassVar

from torch.optim import Adam

from dev_misc import add_argument, g
from dev_misc.devlib.helper import has_gpus
from dev_misc.trainlib import BaseSetting
from sound_law.data.data_loader import DataLoaderRegistry
from sound_law.data.dataset import Alphabet, Split, get_paths
from sound_law.evaluate.evaluator import Evaluator
from sound_law.model.one_pair import OnePairModel
from sound_law.model.one_to_many import OneToManyModel

from .trainer import Trainer

add_argument('check_interval', default=10, dtype=int, msg='Frequency to check the training progress.')
add_argument('eval_interval', default=100, dtype=int, msg='Frequency to call the evaluator.')


class OnePairTask(BaseSetting):
    # TODO(j_luo) Use Setting instead of task so that split or other metadata can be included here as well.
    name: ClassVar[str] = 'one_pair'


@dataclass
class Setting(BaseSetting):
    task: str
    split: Split
    src_lang: str
    tgt_lang: str
    src_abc: Alphabet
    tgt_abc: Alphabet


class OnePairManager:
    """A manager for sound law induction on one src-tgt pair."""

    def __init__(self):
        # Prepare alphabets first.
        src_path, tgt_path = get_paths(g.data_path, g.src_lang, g.tgt_lang)
        self.src_abc = Alphabet.from_tsv(g.src_lang, src_path, g.input_format)
        self.tgt_abc = Alphabet.from_tsv(g.tgt_lang, tgt_path, g.input_format)

        # Prepare data loaders with different splits.
        self.dl_reg = DataLoaderRegistry()
        # self.tasks = dict()  # TODO(j_luo) Integrate tasks into dl_reg.

        def create_setting(name: str, split: Split) -> Setting:
            return Setting(name, 'one_pair', split, g.src_lang, g.tgt_lang, self.src_abc, self.tgt_abc)

        if g.input_format == 'wikt':
            # self.tasks['train'] = train_task = OnePairTask()
            # self.tasks['dev'] = dev_task = OnePairTask()
            train_setting = create_setting('train', Split('train'))
            dev_setting = create_setting('dev', Split('dev'))
            self.dl_reg.register_data_loader(train_setting)
            self.dl_reg.register_data_loader(dev_setting)
            # train_dl = self.dl_reg.register_data_loader(
            #     train_task, train_split, g.src_lang, g.tgt_lang, self.src_abc, self.tgt_abc)
            # dev_dl = self.dl_reg.register_data_loader(
            #     dev_task, dev_split, g.src_lang, g.tgt_lang, self.src_abc, self.tgt_abc)
        else:
            for fold in range(5):
                dev_fold = fold + 1
                train_folds = list(range(1, 6))
                train_folds.remove(dev_fold)

                train_setting = create_setting(f'train@{fold}', Split('train', train_folds))
                dev_setting = create_setting(f'dev@{fold}', Split('dev', [dev_fold]))
                self.dl_reg.register_data_loader(train_setting)
                self.dl_reg.register_data_loader(dev_setting)

                # self.tasks[f'train@{fold}'] = train_task = OnePairTask()
                # self.tasks[f'dev@{fold}'] = dev_task = OnePairTask()

                # train_dl = self.dl_reg.register_data_loader(
                #     train_task, train_split, g.src_lang, g.tgt_lang, self.src_abc, self.tgt_abc)
                # dev_dl = self.dl_reg.register_data_loader(
                #     dev_task, dev_split, g.src_lang, g.tgt_lang, self.src_abc, self.tgt_abc)
        # self.tasks['test'] = test_task = OnePairTask()
        # test_split = Split('test')
        # test_dl = self.dl_reg.register_data_loader(
        #     test_task, test_split, g.src_lang, g.tgt_lang, self.src_abc, self.tgt_abc)
        test_setting = create_setting('test', Split('test'))
        self.dl_reg.register_data_loader(test_setting)

    def run(self):

        def run_once(train_name, dev_name, test_name):
            train_dl = self.dl_reg[train_name]
            dev_dl = self.dl_reg[dev_name]
            test_dl = self.dl_reg[test_name]

            model = OnePairModel(len(self.src_abc), len(self.tgt_abc))
            if has_gpus():
                model.cuda()
            evaluator = Evaluator(model, {train_name: train_dl, dev_name: dev_dl, test_name: test_dl})

            trainer = Trainer(model, [self.dl_reg.get_setting_by_name(train_name)],
                              [1.0], 'step',
                              check_interval=g.check_interval,
                              evaluator=evaluator,
                              eval_interval=g.eval_interval)
            trainer.init_params('uniform', -0.1, 0.1)
            trainer.set_optimizer(Adam, lr=0.002)
            trainer.train(self.dl_reg)

        # test_task = self.tasks['test']
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
        # eval_dls = dict()

        # Get the test language.
        # tgt_task = OnePairTask()
        # tgt_split = Split('all')  # Use the entire dataset for testing.
        # eval_dls[f'test@{g.tgt_lang}'] = test_dl = self.dl_reg.register_data_loader(
        #     tgt_task, tgt_split, g.src_lang, g.tgt_lang, self.src_abc, self.tgt_abc, lang2id=lang2id)

        def create_setting(name: str, tgt_lang: str, split: Split) -> Setting:
            return Setting(name, 'one_pair', split, g.src_lang, tgt_lang, self.src_abc, self.tgt_abc)

        test_setting = create_setting(f'test@{g.tgt_lang}', g.tgt_lang, Split('all'))
        self.dl_reg.register_data_loader(test_setting, lang2id=lang2id)

        # Get the training languages.
        for train_tgt_lang in g.train_tgt_langs:
            # train_task = OnePairTask()
            # train_tasks.append(train_task)
            if g.input_format == 'ielex':
                train_split = Split('train', [1, 2, 3, 4])  # Use the first four folds for training.
                dev_split = Split('dev', [5])  # Use the last fold for dev.
            else:
                train_split = Split('train')
                dev_split = Split('dev')
            train_setting = create_setting(f'train@{train_tgt_lang}', train_tgt_lang, train_split)
            dev_setting = create_setting(f'dev@{train_tgt_lang}', train_tgt_lang, dev_split)
            test_setting = create_setting(f'test@{train_tgt_lang}', train_tgt_lang, Split('test'))

            # dev_task = OnePairTask()

            self.dl_reg.register_data_loader(train_setting, lang2id=lang2id)
            self.dl_reg.register_data_loader(dev_setting, lang2id=lang2id)
            self.dl_reg.register_data_loader(test_setting, lang2id=lang2id)

            # test_task = OnePairTask()
            # test_split = Split('test')

            # eval_dls[f'train@{train_tgt_lang}'] = self.dl_reg.register_data_loader(train_task, train_split, g.src_lang, train_tgt_lang,
            #                                                              self.src_abc, self.tgt_abc, lang2id=lang2id)
            # eval_dls[f'dev@{train_tgt_lang}'] = self.dl_reg.register_data_loader(
            #     dev_task, dev_split, g.src_lang, train_tgt_lang, self.src_abc, self.tgt_abc, lang2id=lang2id)
            # eval_dls[f'test@{train_tgt_lang}'] = self.dl_reg.register_data_loader(test_task, test_split, g.src_lang, train_tgt_lang,
            #                                                             self.src_abc, self.tgt_abc, lang2id=lang2id)

        self.model = OneToManyModel(len(self.src_abc), len(self.tgt_abc), len(g.train_tgt_langs) + 1)
        if has_gpus():
            self.model.cuda()

        # NOTE(j_luo) Evaluate on every loader.
        eval_dls = self.dl_reg.get_loaders_by_name(lambda name: True)
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

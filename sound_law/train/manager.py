"""
A manager class takes care of managing the data loader, the model, and the trainer.
"""
import logging
from typing import ClassVar

from torch.optim import Adam

from dev_misc import add_argument, g
from dev_misc.devlib.helper import has_gpus
from dev_misc.trainlib import Task
from sound_law.data.data_loader import DataLoaderRegistry
from sound_law.data.dataset import Alphabet, Split, get_paths
from sound_law.evaluate.evaluator import Evaluator
from sound_law.model.one_pair import OnePairModel
from sound_law.model.one_to_many import OneToManyModel

from .trainer import Trainer

add_argument('check_interval', default=10, dtype=int, msg='Frequency to check the training progress.')
add_argument('eval_interval', default=100, dtype=int, msg='Frequency to call the evaluator.')


class OnePairTask(Task):
    name: ClassVar[str] = 'one_pair'


class OnePairManager:
    """A manager for sound law induction on one src-tgt pair."""

    def __init__(self):
        # Prepare alphabets first.
        src_path, tgt_path = get_paths(g.data_path, g.src_lang, g.tgt_lang)
        self.src_abc = Alphabet.from_tsv(g.src_lang, src_path)
        self.tgt_abc = Alphabet.from_tsv(g.tgt_lang, tgt_path)

        # Prepare data loaders with different splits.
        self.dl_reg = DataLoaderRegistry()
        self.tasks = dict()  # TODO(j_luo) Integrate tasks into dl_reg.
        for fold in range(5):
            dev_fold = fold + 1
            train_folds = list(range(1, 6))
            train_folds.remove(dev_fold)

            train_split = Split('train', train_folds)
            dev_split = Split('dev', [dev_fold])

            self.tasks[f'train@{fold}'] = train_task = OnePairTask()
            self.tasks[f'dev@{fold}'] = dev_task = OnePairTask()

            train_dl = self.dl_reg.register_data_loader(train_task, train_split, self.src_abc, self.tgt_abc)
            dev_dl = self.dl_reg.register_data_loader(dev_task, dev_split, self.src_abc, self.tgt_abc)
        test_task = OnePairTask()
        test_split = Split('test')
        test_dl = self.dl_reg.register_data_loader(test_task, test_split, self.src_abc, self.tgt_abc)

    def run(self):
        for fold in range(5):
            logging.imp(f'Cross-validation, fold number {fold}')
            train_task = self.tasks[f'train@{fold}']
            dev_task = self.tasks[f'dev@{fold}']

            train_dl = self.dl_reg[train_task]
            dev_dl = self.dl_reg[dev_task]

            model = OnePairModel(len(self.src_abc), len(self.tgt_abc))
            if has_gpus():
                model.cuda()
            evaluator = Evaluator(model, {f'train@{fold}': train_dl, f'dev@{fold}': dev_dl})

            trainer = Trainer(model, [train_task],
                              [1.0], 'step',
                              check_interval=g.check_interval,
                              evaluator=evaluator,
                              eval_interval=g.eval_interval)
            trainer.init_params(method='xavier_uniform')
            trainer.set_optimizer(Adam, lr=0.002)
            trainer.train(self.dl_reg)


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
        self.src_abc = Alphabet.from_tsvs(g.src_lang, src_paths)
        self.tgt_abc = Alphabet.from_tsvs('all_targets', tgt_paths)

        # Get language-to-id mappings. Used only for the targets (i.e., decoder side).
        lang2id = {tgt: i for i, tgt in enumerate(all_tgt)}

        # Get all data loaders.
        self.dl_reg = DataLoaderRegistry()
        self.tasks = dict()
        # Get the test language.
        self.tasks[f'test@{g.tgt_lang}'] = test_task = OnePairTask()
        tgt_split = Split('all')  # Use the entire dataset for testing.
        test_dl = self.dl_reg.register_data_loader(test_task, tgt_split, self.src_abc, self.tgt_abc, lang2id=lang2id)
        # Get the training languages.
        eval_dls = dict()
        for lang in g.train_tgt_langs:
            self.tasks[f'train@{lang}'] = train_task = OnePairTask()
            train_split = Split('train', [1, 2, 3, 4])  # Use the first four folds for training.
            self.tasks[f'dev@{lang}'] = dev_task = OnePairTask()
            dev_split = Split('dev', [5])  # Use the last fold for dev.

            self.dl_reg.register_data_loader(train_task, train_split, self.src_abc, self.tgt_abc, lang2id=lang2id)
            eval_dls[f'dev@{lang}'] = self.dl_reg.register_data_loader(
                dev_task, dev_split, self.src_abc, self.tgt_abc, lang2id=lang2id)

        self.model = OneToManyModel(len(self.src_abc), len(self.tgt_abc), len(g.train_tgt_langs))
        if has_gpus():
            self.model.cuda()

        self.evaluator = Evaluator(self.model, eval_dls)

        train_tasks = [self.tasks[f'train@{lang}'] for lang in g.train_tgt_langs]
        self.trainer = Trainer(self.model, train_tasks,
                               [1.0] * len(train_tasks), 'step',
                               evaluator=self.evaluator,
                               check_interval=g.check_interval,
                               eval_interval=g.eval_interval)

    def run(self):
        self.trainer.train()

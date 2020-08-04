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
from sound_law.data.dataset import Alphabet, Split
from sound_law.evaluate.evaluator import OnePairEvaluator
from sound_law.model.one_pair import OnePairModel

from .trainer import OnePairTrainer

add_argument('check_interval', default=10, dtype=int, msg='Frequency to check the training progress.')
add_argument('eval_interval', default=100, dtype=int, msg='Frequency to call the evaluator.')


class OnePairTask(Task):
    name: ClassVar[str] = 'one_pair'


class OnePairManager:
    """A manager for sound law induction on one src-tgt pair."""

    def __init__(self):
        # Prepare alphabets first.
        src_path = g.data_path / f'{g.src_lang}.tsv'
        self.src_abc = Alphabet.from_tsv(g.src_lang, src_path)
        tgt_path = g.data_path / f'{g.tgt_lang}.tsv'
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
            evaluator = OnePairEvaluator(model, {f'train@{fold}': train_dl, f'dev@{fold}': dev_dl})

            trainer = OnePairTrainer(model, [train_task],
                                     [1.0], 'step',
                                     check_interval=g.check_interval,
                                     evaluator=evaluator,
                                     eval_interval=g.eval_interval)
            trainer.init_params(method='xavier_uniform')
            trainer.set_optimizer(Adam, lr=0.002)
            trainer.train(self.dl_reg)
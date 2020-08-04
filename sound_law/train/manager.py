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
from sound_law.data.dataset import Split
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

            train_dl = self.dl_reg.register_data_loader(train_task, train_split)
            dev_dl = self.dl_reg.register_data_loader(dev_task, dev_split)
        test_task = OnePairTask()
        test_split = Split('test')
        test_dl = self.dl_reg.register_data_loader(test_task, test_split)

        # For consistency, use the entire dataset to init the model.
        # FIXME(j_luo) alphabet should be used for init everything for consistent manppings.
        all_task = OnePairTask()
        all_split = Split('all')
        all_dl = self.dl_reg.register_data_loader(all_task, all_split)
        num_src_chars = len(all_dl.dataset.src_abc)
        num_tgt_chars = len(all_dl.dataset.tgt_abc)
        # FIXME(j_luo) Move model to run.
        self.model = OnePairModel(num_src_chars, num_tgt_chars)
        if has_gpus():
            self.model.cuda()

    def run(self):
        for fold in range(5):
            logging.imp(f'Cross-validation, fold number {fold}')
            train_task = self.tasks[f'train@{fold}']
            dev_task = self.tasks[f'dev@{fold}']

            train_dl = self.dl_reg[train_task]
            dev_dl = self.dl_reg[dev_task]
            evaluator = OnePairEvaluator(self.model, {f'train@{fold}': train_dl, f'dev@{fold}': dev_dl})

            trainer = OnePairTrainer(self.model, [train_task],
                                     [1.0], 'step',
                                     check_interval=g.check_interval,
                                     evaluator=evaluator,
                                     eval_interval=g.eval_interval)
            trainer.init_params(method='xavier_uniform')
            trainer.set_optimizer(Adam, lr=0.002)
            trainer.train(self.dl_reg)

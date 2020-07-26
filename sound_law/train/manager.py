"""
A manager class takes care of managing the data loader, the model, and the trainer.
"""
from typing import ClassVar

from torch.optim import Adam

from dev_misc import g
from dev_misc.trainlib import Task
from sound_law.data.data_loader import DataLoaderRegistry
from sound_law.model.one_pair import OnePairModel

from .trainer import OnePairTrainer


class OnePairTask(Task):
    name: ClassVar[str] = 'one_pair'


class OnePairManager:
    """A manager for sound law induction on one src-tgt pair."""

    def __init__(self):
        self.dl_reg = DataLoaderRegistry()
        one_pair_task = OnePairTask()
        one_pair_dl = self.dl_reg.register_data_loader(one_pair_task)
        num_src_chars = len(one_pair_dl.dataset.src_abc)
        num_tgt_chars = len(one_pair_dl.dataset.tgt_abc)
        self.model = OnePairModel(num_src_chars, num_tgt_chars)
        if g.gpus is not None:  # FIXME(j_luo) Probably need some other variable
            self.model.cuda()
        self.trainer = OnePairTrainer(self.model, [one_pair_task], [1.0], 'step', check_interval=100)
        self.trainer.set_optimizer(Adam, lr=0.002)

    def run(self):
        self.trainer.train(self.dl_reg)

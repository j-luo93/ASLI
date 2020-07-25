"""
A manager class takes care of managing the data loader, the model, and the trainer.
"""
from typing import ClassVar

from dev_misc.trainlib import Task
from sound_law.data.data_loader import DataLoaderRegistry
from .trainer import MonoTrainer
from sound_law.model.mono import MonoModel


class MonoTask(Task):
    name: ClassVar[str] = 'mono'


class MonoManager:
    """A manager for monolingual sound law induction."""

    def __init__(self):
        dl_reg = DataLoaderRegistry()
        mono_task = MonoTask()
        mono_dl = dl_reg.register_data_loader(mono_task)
        num_src_chars = ...  # FIXME(j_luo) fill in this
        num_tgt_chars = ...  # FIXME(j_luo) fill in this
        model = MonoModel(num_src_chars, num_tgt_chars)
        trainer = MonoTrainer(model, [mono_task], [1.0], 'step')  # FIXME(j_luo) fill in this

    def run(self):
        trainer.run()

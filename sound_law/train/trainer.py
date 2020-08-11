import logging

import torch
from torch.nn.utils import clip_grad_norm_

from dev_misc import add_argument, g
from dev_misc.trainlib import Metric, Metrics
from dev_misc.trainlib.base_trainer import BaseTrainer
from sound_law.data.data_loader import OnePairDataLoader


class Trainer(BaseTrainer):

    add_argument('num_steps', default=1000, dtype=int, msg='Number of steps for training.')
    add_argument('save_model', dtype=bool, default=True, msg='Flag to save model.')

    def add_trackables(self):
        self.tracker.add_count_trackable('step', g.num_steps)

    def save(self, eval_metrics: Metrics):
        if g.save_model:
            path = g.log_dir / 'saves' / f'model.{self.stage}.pth'
            path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(self.model.state_dict(), path)
            logging.info(f'Model saved to {path}.')
        else:
            logging.info('No model is saved.')

    def train_one_step(self, dl: OnePairDataLoader) -> Metrics:
        batch = dl.get_next_batch()

        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model(batch)
        # loss = Metric('loss', loss.sum(), batch.num_tgt_units)
        loss = Metric('loss', loss.sum(), len(batch))  # batch.num_tgt_units)
        loss.mean.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), 5.0)
        grad_norm = Metric('grad_norm', grad_norm, len(batch))
        self.optimizer.step()

        return Metrics(loss, grad_norm)

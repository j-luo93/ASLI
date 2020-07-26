from torch.nn.utils import clip_grad_norm_

from dev_misc import add_argument, g
from dev_misc.trainlib import Metric, Metrics
from dev_misc.trainlib.base_trainer import BaseTrainer
from sound_law.data.data_loader import OnePairDataLoader


class OnePairTrainer(BaseTrainer):

    add_argument('num_steps', default=1000, dtype=int, msg='Number of steps for training.')

    def add_trackables(self):
        self.tracker.add_count_trackable('step', g.num_steps)

    def save(self, eval_metrics: Metrics):
        pass

    def train_one_step(self, dl: OnePairDataLoader) -> Metrics:
        batch = dl.get_next_batch()

        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss = Metric('loss', loss.sum(), len(batch))  # FIXME(j_luo) fill in this: fix the padding
        loss.mean.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), 5.0)
        grad_norm = Metric('grad_norm', grad_norm, len(batch))
        self.optimizer.step()

        return Metrics(loss, grad_norm)

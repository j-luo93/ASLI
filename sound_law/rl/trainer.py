from torch.nn.utils import clip_grad_norm_

from dev_misc import g
from dev_misc.trainlib import Metric, Metrics, init_params
from dev_misc.trainlib.base_trainer import BaseTrainer
from sound_law.data.data_loader import (OnePairBatch, OnePairDataLoader,
                                        PaddedUnitSeqs)

from .data_loader import EntireBatchOnePairDataLoader
from .env import SoundChangeEnv, TrajectoryCollector
from .model import ActorCritic
from .trajectory import VocabState


class ActorCriticTrainer(BaseTrainer):
    # FIXME(j_luo) This should be merged later on with `Trainer` in `sound_law`.
    model: ActorCritic
    collector: TrajectoryCollector

    def __init__(self, *args, collector: TrajectoryCollector = None, env: SoundChangeEnv = None, **kwargs):
        if collector is None:
            raise TypeError(f'Must pass a trajectory collector to initialize this trainer.')
        if env is None:
            raise TypeError(f'Must pass an environment to initialize this trainer.')

        self.collector = collector
        self.env = env
        super().__init__(*args, **kwargs)

    def add_trackables(self):
        # FIXME(j_luo) STEP
        self.tracker.add_count_trackable('STEP', g.num_steps)

    def save(self, eval_metrics: Metrics):
        # FIXME(j_luo)
        ...

    def train_one_step(self, dl: EntireBatchOnePairDataLoader) -> Metrics:
        batch: OnePairBatch = dl.get_next_batch()
        init_state = VocabState.from_seqs(batch.src_seqs)
        end_state = VocabState.from_seqs(batch.tgt_seqs)

        # FIXME(j_luo) rename model to agent?
        self.model.train()
        self.optimizer.zero_grad()

        agent_inputs = self.collector.collect(self.model, self.env, init_state, end_state)
        metrics = self.model(agent_inputs)
        metrics.loss.mean.backward()

        # Clip gradient norm.
        grad_norm = clip_grad_norm_(self.model.parameters(), 5.0)
        grad_norm = Metric('grad_norm', grad_norm, agent_inputs.batch_size)
        metrics += grad_norm

        # Update.
        self.optimizer.step()

        metrics = metrics.with_prefix('check')
        return metrics

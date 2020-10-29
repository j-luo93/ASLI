import logging

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from dev_misc import FT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import get_named_range
from dev_misc.trainlib import Metric, Metrics, init_params
from dev_misc.trainlib.base_trainer import BaseTrainer as BaseTrainerDev
from sound_law.data.alphabet import Alphabet
from sound_law.data.data_loader import (OnePairBatch, OnePairDataLoader,
                                        PaddedUnitSeqs, VSOnePairDataLoader)
from sound_law.evaluate.edit_dist import edit_dist_batch
from sound_law.model.decoder import get_beam_probs
from sound_law.rl.agent import AgentInputs, AgentOutputs, BasePG
from sound_law.rl.env import SoundChangeEnv, TrajectoryCollector
from sound_law.rl.trajectory import VocabState


def get_ce_loss(log_probs: FT, batch: OnePairBatch, agg='all') -> FT:
    ce_losses = -log_probs.gather('unit', batch.tgt_seqs.ids)
    weights = batch.tgt_seqs.paddings.float()
    ce_losses = ce_losses * weights
    if agg == 'batch':
        return ce_losses.sum(dim='pos')
    elif agg == 'batch_mean':
        return ce_losses.sum(dim='pos') / weights.sum(dim='pos')
    elif agg == 'all':
        return ce_losses.sum()
    elif agg == 'char':
        return ce_losses
    elif agg == 'char_mean':
        return ce_losses.sum() / weights.sum()
    else:
        raise ValueError(f'Unrecognized value "{agg}" for agg.')


class BaseTrainer(BaseTrainerDev):

    add_argument('num_steps', default=1000, dtype=int, msg='Number of steps for training.')
    add_argument('save_model', dtype=bool, default=True, msg='Flag to save model.')
    add_argument('almt_reg_hyper', dtype=float, default=0.0, msg='Hypagentgularization.')
    add_argument('concentration_scale', dtype=float, default=1.0, msg='Hyperparameter for concentration scale.')
    add_argument('train_mode', dtype=str, default='mle',
                 choices=['mle', 'mrt'], msg='Training mode: either MRT or MLE.')

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


class Trainer(BaseTrainer):

    def _train_one_step_mle(self, batch: OnePairBatch) -> Metrics:
        """Train for one step using maximum likelihood."""
        log_probs, almt_distrs = self.model(batch)

        metrics = Metrics()
        # Cross-entropy loss.
        ce_loss = get_ce_loss(log_probs, batch, agg='all')
        ce_loss = Metric('ce_loss', ce_loss, len(batch))
        metrics += ce_loss

        # Compute alignment regularization loss if needed.
        if g.almt_reg_hyper > 0:
            sl = almt_distrs.size("src_pos")
            pos = get_named_range(sl, 'src_pos').float()
            mean_pos = (pos.align_as(almt_distrs) * almt_distrs).sum(dim='src_pos')
            mean_pos = mean_pos.align_to('batch', 'tgt_pos')
            mean_pos = torch.cat([get_zeros(len(batch), 1), mean_pos], dim=-1)
            src_lengths = batch.src_seqs.lengths.float().rename(None)
            reg_weight = src_lengths.unsqueeze(dim=-1) - 1.0 - mean_pos[:, :-1]
            reg_weight.clamp_(0.0, 1.0)
            rel_pos = mean_pos[:, 1:] - mean_pos[:, :-1]  # bs x tl
            rel_pos_diff = rel_pos - 1
            margin = rel_pos_diff != 0
            almt_reg = margin.float() * (rel_pos_diff ** 2)  # bs x tl
            almt_reg = (almt_reg * reg_weight).sum()
            almt_reg = Metric('almt_reg', almt_reg, len(batch))
            metrics += almt_reg

            loss = ce_loss.mean + g.almt_reg_hyper * almt_reg.mean
        else:
            loss = ce_loss.mean
        loss = Metric('loss', loss * len(batch), len(batch))
        metrics += loss
        return metrics

    def _train_one_step_mrt(self, batch: OnePairBatch, abc: Alphabet) -> Metrics:
        """Train for one step using minimum risk training (MRT)."""
        # Get scores for top predictions and the target sequence.
        assert g.comp_mode == 'str'
        assert g.dropout == 0.0, 'Might have some issue due to the fact that ground truth is fed through the normal forward call whereas hyps are not, resulting in discrepant dropout applications.'
        hyps = self.model.predict(batch)
        log_probs, _ = self.model(batch)
        tgt_scores = -get_ce_loss(log_probs, batch, agg='batch')

        # Mark which ones are duplicates.
        preds, _, _ = hyps.translate(abc)
        duplicates = list()
        for beam_preds, tgt_form in zip(preds, batch.tgt_seqs.forms):
            duplicates.append([False] + [p == tgt_form for p in beam_preds])
            # If no duplicates are found, then we discard the last prediction.
            if not any(duplicates[-1]):
                duplicates[-1][-1] = True
            if sum(duplicates[-1]) != 1:
                raise RuntimeError(f'Should have exactly one duplicate.')
        duplicates = get_tensor(duplicates)

        # Assemble all scores together.
        scores = torch.cat([tgt_scores.align_as(hyps.scores), hyps.scores], dim='beam')
        probs = get_beam_probs(scores, duplicates=duplicates)
        target = np.tile(batch.tgt_seqs.forms.reshape(-1, 1), [1, g.beam_size + 1])
        preds = np.concatenate([target[:, 0:1], preds], axis=-1)
        dists = edit_dist_batch(preds.reshape(-1), target.reshape(-1), 'ed')
        lengths = batch.tgt_seqs.lengths.align_to('batch', 'beam')
        dists = get_tensor(dists.reshape(-1, g.beam_size + 1)).float()  # / lengths
        # risk = (probs * (dists ** 2)).sum(dim='beam')
        risk = (probs * dists).sum(dim='beam')
        risk = Metric('risk', risk.sum(), len(batch))
        return Metrics(risk)

    def train_one_step(self, dl: OnePairDataLoader) -> Metrics:
        batch: OnePairBatch = dl.get_next_batch()

        self.model.train()
        self.optimizer.zero_grad()

        if g.train_mode == 'mrt':
            metrics = self._train_one_step_mrt(batch, dl.tgt_abc)
            metrics.risk.mean.backward()
        else:
            metrics = self._train_one_step_mle(batch)
            metrics.loss.mean.backward()

        # Clip gradient norm.
        grad_norm = clip_grad_norm_(self.model.parameters(), 5.0)
        grad_norm = Metric('grad_norm', grad_norm, len(batch))
        metrics += grad_norm

        # Update.
        self.optimizer.step()

        metrics = metrics.with_prefix('check')
        return metrics


def log_trajectories(agent_inputs: AgentInputs, n: int = 5):
    for i, tr in enumerate(agent_inputs.trajectories[:n], 1):
        logging.info(f'Sample trajectory {i}')
        logging.info(str(tr))


class PolicyGradientTrainer(BaseTrainer):

    model: BasePG
    collector: TrajectoryCollector

    add_argument('entropy_reg', dtype=float, default=0.0, msg='Entropy regularization hyperparameter.')
    add_argument('value_steps', dtype=int, default=0, msg='How many inner loops to fit value net.')

    def add_trackables(self):
        super().add_trackables()
        if g.value_steps:
            self.tracker.add_trackable('value_step', total=g.value_steps, endless=True)
            self.tracker.add_trackable('policy_step', total=1, endless=True)

    def __init__(self, *args, collector: TrajectoryCollector = None, env: SoundChangeEnv = None, **kwargs):
        if collector is None:
            raise TypeError(f'Must pass a trajectory collector to initialize this trainer.')
        if env is None:
            raise TypeError(f'Must pass an environment to initialize this trainer.')

        self.collector = collector
        self.env = env
        super().__init__(*args, **kwargs)

    @property
    def agent(self) -> BasePG:
        return self.model

    def train_one_step(self, dl: VSOnePairDataLoader) -> Metrics:
        init_state = dl.init_state
        end_state = dl.end_state

        # Collect episodes first.
        agent_inputs = self.collector.collect(self.agent, self.env, init_state, end_state)
        self.add_callback('check', 'log_tr', lambda: log_trajectories(agent_inputs))
        bs = agent_inputs.batch_size
        n_tr = len(agent_inputs.trajectories)

        # ---------------------------- main ---------------------------- #

        def get_v_loss(agent_outputs: AgentOutputs) -> Metric:
            rew_outputs = agent_outputs.rew_outputs
            diff = (rew_outputs.values - rew_outputs.rtgs) ** 2
            loss = Metric('v_regress_loss', 0.5 * diff.sum(), len(rew_outputs.values))
            return loss

        def get_pi_losses(agent_outputs: AgentOutputs) -> Metrics:
            log_probs = agent_outputs.log_probs
            entropy = agent_outputs.entropy
            rtgs = agent_outputs.rew_outputs.rtgs
            pg_losses = (-log_probs * rtgs)
            pg = Metric('pg', pg_losses.sum(), bs)
            entropy_loss = Metric('entropy', entropy.sum(), bs)
            pi_loss = Metric('pi_loss', pg.total - g.entropy_reg * entropy_loss.total, bs)
            return Metrics(pg, entropy_loss, pi_loss)

        def get_optim_params(optim):
            for param_group in optim.param_groups:
                yield from param_group['params']

        def update(name: str) -> Metrics:
            self.model.train()
            if name == 'all':
                # Use the default optimizer that optimize the entire model.
                optim = self.optimizer
            else:
                optim = self.optimizers[name]
            optim.zero_grad()

            ret_log_probs = ret_entropy = (name != 'value')
            agent_outputs: AgentOutputs = self.model(agent_inputs,
                                                     ret_log_probs=ret_log_probs,
                                                     ret_entropy=ret_entropy)

            # Some common metrics.
            tr_rew = Metric('reward', agent_inputs.rewards.sum(), n_tr)
            success = Metric('success', agent_inputs.done.sum(), n_tr)
            step_metrics = Metrics(tr_rew, success)

            # Compute losses depending on the name.
            if name in ['value', 'all'] and g.agent == 'a2c':
                step_metrics += get_v_loss(agent_outputs)
            if name in ['policy', 'all']:
                step_metrics += get_pi_losses(agent_outputs)
                if g.agent == 'a2c':
                    abs_advs = agent_outputs.rew_outputs.advantages.abs()
                    step_metrics += Metric('abs_advantage', abs_advs.sum(), bs)

            # Backprop depending on the name.
            if name == 'value':
                step_metrics.v_regress_loss.mean.backward()
            elif name == 'policy':
                step_metrics.pi_loss.mean.backward()
            else:
                total_loss = step_metrics.pi_loss.total + step_metrics.v_regress_loss.total
                total_loss = Metric('total_loss', total_loss, bs)
                step_metrics += total_loss
                step_metrics.total_loss.mean.backward()

            # Clip gradient norm.
            grad_norm = clip_grad_norm_(get_optim_params(optim), 5.0)
            grad_norm = Metric('grad_norm', grad_norm, bs)
            step_metrics += grad_norm

            # Update.
            optim.step()

            return step_metrics

        # Gather metrics.
        metrics = Metrics()
        if g.value_steps and g.agent == 'a2c':
            with self.model.policy_grad(True), self.model.value_grad(False):
                metrics += update('policy')
                self.tracker.update('policy_step')

            with self.model.policy_grad(False), self.model.value_grad(True):
                for _ in range(g.value_steps):
                    metrics += update('value')
                    self.tracker.update('value_step')
        else:
            name = 'all' if g.agent == 'a2c' else 'policy'
            with self.model.policy_grad(True), self.model.value_grad(True):
                metrics += update(name)

        metrics = metrics.with_prefix('check')
        return metrics

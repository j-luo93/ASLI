import logging

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from dev_misc import FT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import get_named_range
from dev_misc.trainlib import Metric, Metrics
from dev_misc.trainlib.base_trainer import BaseTrainer
from sound_law.data.alphabet import Alphabet
from sound_law.data.data_loader import OnePairBatch, OnePairDataLoader
from sound_law.evaluate.edit_dist import edit_dist_batch
from sound_law.model.decoder import get_beam_probs


def _get_ce_loss(log_probs: FT, batch: OnePairBatch, mean='all') -> FT:
    ce_losses = -log_probs.gather('unit', batch.tgt_seqs.ids)
    ce_losses = ce_losses * batch.tgt_seqs.paddings.float()
    if mean == 'batch':
        return ce_losses.sum(dim='pos')
    elif mean == 'all':
        return ce_losses.sum()
    else:
        raise ValueError(f'Unrecognized value "{mean}" for mean.')


class Trainer(BaseTrainer):

    add_argument('num_steps', default=1000, dtype=int, msg='Number of steps for training.')
    add_argument('save_model', dtype=bool, default=True, msg='Flag to save model.')
    add_argument('almt_reg_hyper', dtype=float, default=0.0, msg='Hyperparameter for alignment regularization.')
    add_argument('concentration_scale', dtype=float, default=10.0, msg='Hyperparameter for concentration scale.')
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

    def _train_one_step_mle(self, batch: OnePairBatch) -> Metrics:
        """Train for one step using maximum likelihood."""
        log_probs, almt_distrs = self.model(batch)

        metrics = Metrics()
        # Cross-entropy loss.
        ce_loss = _get_ce_loss(log_probs, batch, mean='all')
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
        hyps = self.model.predict(batch)
        log_probs, _ = self.model(batch)
        tgt_scores = -_get_ce_loss(log_probs, batch, mean='batch')

        # Mark which ones are duplicates.
        preds, _ = hyps.translate(abc)
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

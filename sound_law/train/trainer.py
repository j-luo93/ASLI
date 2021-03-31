import logging
import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch

from dev_misc import FT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName, get_named_range
from dev_misc.trainlib import (Metric, Metrics, clip_grad, get_optim_params,
                               init_params)
from dev_misc.trainlib.base_trainer import BaseTrainer as BaseTrainerDev
from dev_misc.utils import pad_for_log
from sound_law.data.alphabet import SENTINEL_ID, Alphabet
from sound_law.data.data_loader import (OnePairBatch, OnePairDataLoader,
                                        PaddedUnitSeqs, VSOnePairDataLoader)
from sound_law.evaluate.edit_dist import edit_dist_batch
from sound_law.rl.agent import AgentInputs, AgentOutputs, BasePG
from sound_law.rl.env import SoundChangeEnv  # , TrajectoryCollector
from sound_law.rl.mcts import Mcts
# pylint: disable=no-name-in-module
from sound_law.rl.reward import get_rtgs
# pylint: enable=no-name-in-module
from sound_law.rl.trajectory import Trajectory, TrEdge
from sound_law.s2s.decoder import get_beam_probs


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
    add_argument('almt_reg_hyper', dtype=float, default=0.0, msg='Hyperparameter for alignment.')
    add_argument('weight_decay', dtype=float, default=0.0, msg='Hyperparameter for weight decay.')
    add_argument('concentration_scale', dtype=float, default=1.0, msg='Hyperparameter for concentration scale.')
    add_argument('train_mode', dtype=str, default='mle',
                 choices=['mle', 'mrt'], msg='Training mode: either MRT or MLE.')
    add_argument('init_entropy_reg', dtype=float, default=0.0, msg='Initial entropy regularization hyperparameter.')
    add_argument('end_entropy_reg', dtype=float, default=0.0, msg='Bound for entropy regularization hyperparameter.')
    add_argument('when_entropy_reg', dtype=int, default=100,
                 msg='When to reach the bound for entropy regularization hyperparameter.')

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
        grad_norm = clip_grad(self.model.parameters(), len(batch))
        metrics += grad_norm

        # Update.
        self.optimizer.step()

        metrics = metrics.with_prefix('check')
        return metrics


def log_trajectories(agent_inputs: AgentInputs, n: int = 5):
    for i, tr in enumerate(agent_inputs.trajectories[:n], 1):
        logging.debug(f'Sample trajectory {i}')
        logging.debug(str(tr))


class RLTrainer(BaseTrainer):

    model: BasePG

    @property
    def agent(self) -> BasePG:
        return self.model


class MctsTrainer(RLTrainer):

    add_argument('num_mcts_sims', default=100, dtype=int, msg='Number of MCTS simulations to run.')
    add_argument('expansion_batch_size', default=10, dtype=int, msg='Batch size for expansion steps.')
    add_argument('mcts_batch_size', default=128, dtype=int, msg='Batch size for optimizing the MCTS agent.')
    add_argument('replay_buffer_size', default=1024, dtype=int, msg='Size for the replay buffer.')
    add_argument('num_episodes', default=10, dtype=int, msg='Number of episodes.')
    add_argument('num_inner_steps', default=10, dtype=int, msg='Number of optimization step per batch.')
    add_argument('episode_check_interval', default=10, dtype=int, msg='Frequency of checking episodes')
    add_argument('regress_lambda', default=0.01, dtype=float, msg='Hyperparameter for regression loss.')
    add_argument('use_value_guidance', default=True, dtype=bool,
                 msg='Flag to use predicted values to guide the search.')
    add_argument('tau', default=0.0, dtype=float, msg='Temperature for sampling episodes.')

    def __init__(self, *args, mcts: Mcts = None, **kwargs):
        if mcts is None:
            raise TypeError(f'Must pass a trajectory collector to initialize this trainer.')

        if g.num_mcts_sims % g.expansion_batch_size > 0:
            raise ValueError(f'`expansion_batch_size should divide `num_mcts_sims`.')

        self.mcts = mcts
        self.replay_buffer: Deque[TrEdge] = deque(maxlen=g.replay_buffer_size)
        self.buffer_weight: Deque[float] = deque(maxlen=g.replay_buffer_size)
        super().__init__(*args, **kwargs)

    def add_trackables(self):
        super().add_trackables()
        step = self.tracker['step']
        episode = step.add_trackable('episode', total=g.num_episodes, endless=True)
        episode.add_trackable('rollout', total=g.max_rollout_length, endless=True)
        episode.add_trackable('mcts', total=g.num_mcts_sims, endless=True)
        step.add_trackable('inner_step', total=g.num_inner_steps, endless=True)

    def train_one_step(self, dl: OnePairDataLoader):
        # Collect episodes with the latest agent first.
        new_tr = self.mcts.collect_episodes(self.mcts.env.start, self.tracker)
        # new_tr = self.mcts.collect_episodes(dl.init_state, dl.end_state, self.tracker)
        tr_rew = Metric('reward', sum(tr.rewards.sum() for tr in new_tr), g.num_episodes)
        tr_len = Metric('trajectory_length', sum(map(len, new_tr)), g.num_episodes)
        success = Metric('success', sum(tr.done for tr in new_tr), g.num_episodes)
        metrics = Metrics(tr_rew, tr_len, success)

        eval_tr = self.mcts.collect_episodes(self.mcts.env.start, self.tracker, num_episodes=1, is_eval=True)[0]
        metrics +=  Metric('eval_reward', eval_tr.total_reward, 1)

        # Add these new episodes to the replay buffer.
        for i, tr in enumerate(new_tr, 1):
            self.metric_writer.add_scalar('episode_reward', tr.rewards.sum(),
                                          global_step=i + self.tracker['step'].value * g.num_episodes)
            # NOTE(j_luo) Use temperature if it's positive.
            if g.tau > 0.0:
                weight = math.exp(tr.total_reward * 10.0)
            else:
                weight = 1.0

            for tr_edge in tr:
                self.replay_buffer.append(tr_edge)
                self.buffer_weight.append(weight)

        weights = np.asarray(self.buffer_weight)
        weights = weights / weights.sum()
        # Main loop.
        with self.agent.policy_grad(True), self.agent.value_grad(True):
            for _ in range(g.num_inner_steps):
                # Get a batch of training trajectories from the replay buffer.
                edge_batch = np.random.choice(self.replay_buffer, p=weights, size=g.mcts_batch_size)
                # edge_batch = np.random.choice(self.replay_buffer, size=g.mcts_batch_size)
                agent_inputs = AgentInputs.from_edges(edge_batch)  # , self.mcts.env)#, sparse=True)

                self.agent.train()
                self.optimizer.zero_grad()

                policies = self.agent.get_policy(agent_inputs.id_seqs, almts=(agent_inputs.almts1, agent_inputs.almts2))
                # values = self.agent.get_values(agent_inputs.id_seqs, steps=agent_inputs.steps)
                with NoName(policies, agent_inputs.permissible_actions):
                    mask = agent_inputs.permissible_actions == SENTINEL_ID
                    pa = agent_inputs.permissible_actions
                    pa = torch.where(mask, torch.zeros_like(pa), pa)
                    logits = policies.gather(2, pa)
                    logits = torch.where(mask, torch.full_like(logits, -9999.9), logits)
                    logits = logits.log_softmax(dim=-1)
                # r_max = agent_inputs.rewards.max()
                # r_min = agent_inputs.rewards.min()
                # weights = (agent_inputs.rewards - r_min) / (r_max - r_min + 1e-8)

                # weights = weights.align_as(pi_ce_losses)
                entropies = (-agent_inputs.mcts_pis * (1e-8 + agent_inputs.mcts_pis).log()).sum(dim=-1)
                pi_ce_losses = (-agent_inputs.mcts_pis * logits).sum(dim=-1) - entropies
                for i in range(6):
                    metrics += Metric(f'entropy_{i}', entropies[:, i].sum(), g.mcts_batch_size)
                    metrics += Metric(f'pi_ce_los_{i}', pi_ce_losses[:, i].sum(), g.mcts_batch_size)

                # v_regress_losses = 0.5 * (values - agent_inputs.qs) ** 2

                # pi_ce_loss = Metric('pi_ce_loss', (weights * pi_ce_losses).sum(), g.mcts_batch_size * 7)
                # mini_weights = get_tensor([1.0, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]).rename('mini').align_as(pi_ce_losses)
                # pi_ce_loss = Metric('pi_ce_loss', (mini_weights * pi_ce_losses).sum(), g.mcts_batch_size * 7)
                pi_ce_loss = Metric('pi_ce_loss', pi_ce_losses.sum(), g.mcts_batch_size * 7)
                # pi_ce_loss = Metric('pi_ce_loss', pi_ce_losses[:, 0].sum(), g.mcts_batch_size)
                # v_regress_loss = Metric('v_regress_loss', v_regress_losses.sum(), g.mcts_batch_size)
                total_loss = pi_ce_loss.total  # + g.regress_lambda * v_regress_loss.total
                total_loss = Metric('total_loss', total_loss, g.mcts_batch_size)

                total_loss.mean.backward()

                # Clip gradient norm.
                grad_norm = clip_grad(self.agent.parameters(), g.mcts_batch_size)
                # metrics += Metrics(total_loss, pi_ce_loss, v_regress_loss, grad_norm)
                metrics += Metrics(total_loss, pi_ce_loss, grad_norm)
                self.optimizer.step()
                self.tracker.update('inner_step')

        return metrics

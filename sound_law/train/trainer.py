from sound_law.rl.env import stack_ids
import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch

from dev_misc import FT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import get_named_range
from dev_misc.trainlib import (Metric, Metrics, clip_grad, get_optim_params,
                               init_params)
from dev_misc.trainlib.base_trainer import BaseTrainer as BaseTrainerDev
from sound_law.data.alphabet import Alphabet
from sound_law.data.data_loader import (OnePairBatch, OnePairDataLoader,
                                        PaddedUnitSeqs, VSOnePairDataLoader)
from sound_law.evaluate.edit_dist import edit_dist_batch
from sound_law.model.decoder import get_beam_probs
from sound_law.rl.agent import AgentInputs, AgentOutputs, BasePG
from sound_law.rl.env import SoundChangeEnv, TrajectoryCollector
from sound_law.rl.mcts import Mcts


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


class PolicyGradientTrainer(RLTrainer):

    add_argument('value_steps', dtype=int, default=10, msg='How many inner loops to fit value net.')
    add_argument('use_ppo', dtype=bool, default=False, msg='Flag to use PPO training.')
    add_argument('policy_steps', dtype=int, default=10, msg='How many inner loops to train policy net. Used for PPO.')
    add_argument('clip_ratio', dtype=float, default=0.2, msg='Clip ratio used for PPO.')
    add_argument('target_kl', dtype=float, default=0.015, msg='Max kl to use.')
    add_argument('entropy_as_reward', dtype=bool, default=False, msg='Flag to add policy entropy to reward.')

    @property
    def entropy_reg(self) -> float:
        if g.init_entropy_reg > 0.0:
            return self.tracker.entropy_reg
        return 0.0

    def add_trackables(self):
        if g.init_entropy_reg > 0.0:
            multiplier = math.exp(math.log(g.end_entropy_reg / g.init_entropy_reg) / g.when_entropy_reg)
            self.tracker.add_anneal_trackable('entropy_reg', g.init_entropy_reg, multiplier, g.end_entropy_reg)
        if g.value_steps:
            policy_steps = g.policy_steps if g.use_ppo else 1
            self.tracker.add_trackable('policy_step', total=policy_steps, endless=True)
            self.tracker.add_trackable('value_step', total=g.value_steps, endless=True)
        super().add_trackables()

    def __init__(self, *args, collector: TrajectoryCollector = None, env: SoundChangeEnv = None, **kwargs):
        if collector is None:
            raise TypeError(f'Must pass a trajectory collector to initialize this trainer.')
        if env is None:
            raise TypeError(f'Must pass an environment to initialize this trainer.')

        self.collector = collector
        self.env = env
        super().__init__(*args, **kwargs)

    def train_one_step(self, dl: VSOnePairDataLoader) -> Metrics:
        init_state = dl.init_state
        end_state = dl.end_state

        # Collect episodes first.
        agent_inputs = self.collector.collect(self.agent, self.env, init_state, end_state)
        # FIXME(j_luo) a bit ugly here.
        self.add_callback('check', 'log_tr', lambda: log_trajectories(agent_inputs))
        bs = agent_inputs.batch_size
        n_tr = len(agent_inputs.trajectories)

        # ---------------------------- main ---------------------------- #

        def get_v_loss(agent_outputs: AgentOutputs, tgt_agent_outputs: AgentOutputs) -> Metric:
            if g.critic_target == 'rtg':
                tgt = tgt_agent_outputs.rew_outputs.rtgs
            else:
                tgt = tgt_agent_outputs.rew_outputs.expected
                if g.entropy_as_reward:
                    tgt = tgt + self.entropy_reg * tgt_agent_outputs.entropy
            diff = (agent_outputs.rew_outputs.values - tgt) ** 2
            loss = Metric('v_regress_loss', 0.5 * diff.sum(), len(tgt))
            return loss

        def get_pi_losses(agent_outputs: AgentOutputs, tgt_agent_outputs: Optional[AgentOutputs] = None) -> Metrics:
            ret = Metrics()

            log_probs = agent_outputs.log_probs
            entropy = agent_outputs.entropy
            if g.agent == 'vpg':
                pg_losses = -log_probs * agent_outputs.rew_outputs.rtgs
            else:
                if g.use_ppo:
                    tgt_log_probs = tgt_agent_outputs.log_probs
                    ratio = (log_probs - tgt_log_probs).exp()
                    tgt_advs = tgt_agent_outputs.rew_outputs.advantages
                    if g.entropy_as_reward:
                        tgt_advs = tgt_advs + self.entropy_reg * entropy
                    low = 1 - g.clip_ratio
                    high = 1 + g.clip_ratio
                    clip_adv = ratio.clamp(low, high) * tgt_advs
                    pg_losses = -torch.min(ratio * tgt_advs, clip_adv)

                    # Extra useful info.
                    with torch.no_grad():
                        approx_kl = tgt_log_probs - log_probs
                        clipped = (ratio > high) | (ratio < low)
                        ret += Metric('clipped', clipped.sum(), bs)
                        ret += Metric('approx_kl', approx_kl.sum(), bs)
                else:
                    pg_losses = -log_probs * agent_outputs.rew_outputs.advantages
                advs = agent_outputs.rew_outputs.advantages
                ret += Metric('abs_advantage', advs.abs().sum(), bs)
                ret += Metric('advantage', advs.sum(), bs)

            pg = Metric('pg', pg_losses.sum(), bs)
            entropy_loss = Metric('entropy', entropy.sum(), bs)
            if g.entropy_as_reward:
                pi_loss = Metric('pi_loss', pg.total, bs)
            else:
                pi_loss = Metric('pi_loss', pg.total - self.entropy_reg * entropy_loss.total, bs)
            ret += Metrics(pg, entropy_loss, pi_loss)
            return ret

        def update(name: str, tgt_agent_outputs: Optional[AgentOutputs] = None) -> Tuple[Metrics, AgentOutputs]:
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
                step_metrics += get_v_loss(agent_outputs, tgt_agent_outputs)
            if name in ['policy', 'all']:
                step_metrics += get_pi_losses(agent_outputs, tgt_agent_outputs=tgt_agent_outputs)

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
            grad_norm = clip_grad(get_optim_params(optim), bs)
            step_metrics += grad_norm

            # Update.
            optim.step()

            return step_metrics, agent_outputs

        # Gather metrics.
        metrics = Metrics()

        if g.agent == 'a2c':

            # PPO training.
            if g.use_ppo:
                with self.model.policy_grad(False), self.model.value_grad(False):
                    tgt_agent_outputs = self.model(agent_inputs, ret_entropy=True)

                with self.model.policy_grad(True), self.model.value_grad(False):
                    self.tracker.reset('policy_step')
                    for i in range(g.policy_steps):
                        step_metrics, _ = update('policy', tgt_agent_outputs=tgt_agent_outputs)
                        metrics += step_metrics
                        self.tracker.update('policy_step')
                        # Early-stop.
                        if step_metrics.approx_kl.mean.item() > g.target_kl:
                            logging.debug(f'Early-stopped at step {i + 1}.')
                            break
            else:
                with self.model.policy_grad(True), self.model.value_grad(False):
                    step_metrics, tgt_agent_outputs = update('policy')
                    metrics += step_metrics
                    self.tracker.update('policy_step')

            with self.model.policy_grad(False), self.model.value_grad(True):
                for _ in range(g.value_steps):
                    metrics += update('value', tgt_agent_outputs=tgt_agent_outputs)[0]
                    self.tracker.update('value_step')

        else:
            name = 'all' if g.agent == 'a2c' else 'policy'
            with self.model.policy_grad(True), self.model.value_grad(True):
                metrics += update(name)[0]

        metrics = metrics.with_prefix('check')
        if g.init_entropy_reg > 0.0:
            self.tracker.update('entropy_reg')
        return metrics


class MctsTrainer(RLTrainer):

    add_argument('num_mcts_sims', default=100, dtype=int, msg='Number of MCTS simulations to run.')
    add_argument('expansion_batch_size', default=10, dtype=int, msg='Batch size for WU-UCT expansion steps.')
    add_argument('num_episodes', default=10, dtype=int, msg='Number of episodes.')
    add_argument('num_inner_steps', default=10, dtype=int, msg='Number of optimization step per batch.')

    def __init__(self, *args, mcts: Mcts = None, **kwargs):
        if mcts is None:
            raise TypeError(f'Must pass a trajectory collector to initialize this trainer.')

        if g.num_mcts_sims % g.expansion_batch_size > 0:
            raise ValueError(f'`expansion_batch_size should divide `num_mcts_sims`.')

        self.mcts = mcts
        super().__init__(*args, **kwargs)

    def add_trackables(self):
        super().add_trackables()
        self.tracker.add_trackable('episode', total=g.num_episodes, endless=True)
        self.tracker.add_trackable('rollout', total=g.max_rollout_length, endless=True)
        self.tracker.add_trackable('mcts', total=g.num_mcts_sims, endless=True)
        self.tracker.add_trackable('inner_step', total=g.num_inner_steps, endless=True)

    # @profile
    def train_one_step(self, dl: OnePairDataLoader):
        samples = list()
        success = 0.0

        # Collect episodes.
        for ei in range(g.num_episodes):
            root = dl.init_state
            # FIXME(j_luo) reset the prior and stuff.
            self.mcts.reset()
            value = self.mcts.expand(root)
            self.mcts.backup(root, value)

            history = list()
            # Episodes have max rollout length.
            for ri in range(g.max_rollout_length):
                depth_limit = g.max_rollout_length - ri
                # Run many simulations before take one action. Simulations take place in batches. Each batch
                # would be evaluated and expanded after batched selection.
                num_batches = g.num_mcts_sims // g.expansion_batch_size
                for _ in range(num_batches):
                    new_states = self.mcts.parallel_select(root, g.expansion_batch_size, depth_limit)
                    values = self.mcts.expand(new_states)
                    for state, value in zip(new_states, values):
                        self.mcts.backup(state, value)
                    self.tracker.update('mcts', incr=g.expansion_batch_size)
                probs, action, new_state = self.mcts.play(root)
                print(action)
                history.append((probs, root))
                root = new_state

                self.tracker.update('rollout')
                if root == dl.end_state:
                    break

            reward = int(root == dl.end_state)
            success += reward
            samples.extend([(probs, state, reward) for probs, state in history])
            self.tracker.update('episode')

        curr_ids = list()
        action_masks = list()
        rewards = list()
        tgt_policies = list()
        for probs, state, reward in samples:
            curr_ids.extend(state.vocab)
            # FIXME(j_luo) check this
            action_masks.append(self.mcts.action_space.get_action_mask(state))
            tgt_policies.append(probs)
            rewards.append(reward)
        # HACK(j_luo) I don't like this function.
        curr_ids = stack_ids(curr_ids, len(rewards), len(state))
        action_masks = get_tensor(np.stack(action_masks, axis=0)).rename('batch', 'action')
        tgt_policies = get_tensor(np.stack(tgt_policies, axis=0)).rename('batch', 'action')
        rewards = get_tensor(rewards).rename('batch')

        bs = rewards.size('batch')
        success = Metric('success', success, bs)
        metrics = Metrics(success)
        for _ in range(g.num_inner_steps):
            self.agent.train()
            self.optimizer.zero_grad()

            with self.agent.policy_grad(True), self.agent.value_grad(True):
                policy = self.agent.get_policy(curr_ids, action_masks)
                values = self.agent.get_values(curr_ids)
            pi_ce_losses = -tgt_policies * policy.logits
            v_regress_losses = 0.5 * (values - rewards) ** 2

            pi_ce_loss = Metric('pi_ce_loss', pi_ce_losses.sum(), bs)
            v_regress_loss = Metric('v_regress_loss', v_regress_losses.sum(), bs)
            print(pi_ce_loss.mean.item(), v_regress_loss.mean.item())
            total_loss = Metric('total_loss', pi_ce_loss.total + v_regress_loss.total, bs)

            total_loss.mean.backward()

            # Clip gradient norm.
            grad_norm = clip_grad(self.agent.parameters(), bs)
            metrics += Metrics(total_loss, pi_ce_loss, v_regress_loss, grad_norm)
            self.optimizer.step()
            self.tracker.update('inner_step')

        return metrics

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from dev_misc import FT, LT, add_argument, g, get_tensor
from dev_misc.devlib import get_array
from dev_misc.devlib.dp import EditDist
from dev_misc.devlib.tensor_x import TensorX as Tx
from dev_misc.trainlib import Metric, Metrics
from dev_misc.trainlib.tb_writer import MetricWriter
from dev_misc.utils import pad_for_log, pbar
from sound_law.data.alphabet import Alphabet
from sound_law.data.data_loader import OnePairBatch, OnePairDataLoader
from sound_law.evaluate.edit_dist import edit_dist_all
from sound_law.rl.mcts import Mcts
from sound_law.s2s.decoder import get_beam_probs
from sound_law.s2s.one_pair import OnePairModel
from sound_law.train.trainer import get_ce_loss

add_argument('eval_mode', dtype=str, default='edit_dist', choices=[
             'prob', 'edit_dist'], msg='Evaluation mode using probabilities or edit distance.')
add_argument('comp_mode', dtype=str, default='str', choices=['ids', 'units', 'str', 'ids_gpu'],
             msg='Comparison mode.')
add_argument('use_phono_edit_dist', dtype=bool, default=False,
             msg='Flag to use phonologically-aware edit distance.')
add_argument('phono_edit_dist_scale', dtype=float, default=1.0,
             msg='Scaling factor for phonological edit distance.')


# TODO(j_luo) This can be refactored. Also, there are too many edit-dist methods.
def compute_edit_dist(comp_mode: str,
                      pred_ids: Optional[LT] = None,
                      lengths: Optional[LT] = None,
                      gold_ids: Optional[LT] = None,
                      forms: Optional[np.ndarray] = None,
                      predictions=None,
                      pred_lengths: Optional[np.ndarray] = None,
                      units: Optional[np.ndarray] = None,
                      pfm=None) -> FT:
    if comp_mode == 'ids_gpu':
        # Prepare tensorx.
        pred_lengths = Tx(get_tensor(pred_lengths), ['pred_batch', 'beam'])
        pred_tokens = Tx(pred_ids, ['pred_batch', 'beam', 'l'])
        # NOTE(j_luo) -1 for removing EOT's.
        tgt_lengths = Tx(lengths, ['tgt_batch']) - 1
        tgt_tokens = Tx(gold_ids, ['tgt_batch', 'l'])

        # Align them to the same names.
        new_names = ['pred_batch', 'beam', 'tgt_batch']
        pred_lengths = pred_lengths.align_to(*new_names)
        pred_tokens = pred_tokens.align_to(*(new_names + ['l']))
        tgt_lengths = tgt_lengths.align_to(*new_names)
        tgt_tokens = tgt_tokens.align_to(*(new_names + ['l']))

        # Expand them to have the same size.
        pred_bs = pred_tokens.size('pred_batch')
        tgt_bs = tgt_tokens.size('tgt_batch')
        pred_lengths = pred_lengths.expand({'tgt_batch': tgt_bs})
        pred_tokens = pred_tokens.expand({'tgt_batch': tgt_bs})
        tgt_tokens = tgt_tokens.expand({'pred_batch': pred_bs, 'beam': g.beam_size})
        tgt_lengths = tgt_lengths.expand({'pred_batch': pred_bs, 'beam': g.beam_size})

        # Flatten names, preparing for DP.
        def flatten(tx):
            return tx.flatten(['pred_batch', 'beam', 'tgt_batch'], 'batch')

        pred_lengths = flatten(pred_lengths)
        pred_tokens = flatten(pred_tokens)
        tgt_lengths = flatten(tgt_lengths)
        tgt_tokens = flatten(tgt_tokens)

        penalty = None
        if g.use_phono_edit_dist:
            x = pfm.rename('src_unit', 'phono_feat')
            y = pfm.rename('tgt_unit', 'phono_feat')
            names = ('src_unit', 'tgt_unit', 'phono_feat')
            diff = x.align_to(*names) - y.align_to(*names)
            penalty = (diff != 0).sum('phono_feat').cuda().float() / g.phono_edit_dist_scale
        dp = EditDist(pred_tokens, tgt_tokens, pred_lengths, tgt_lengths, penalty=penalty)
        dp.run()
        dists = dp.get_results().data
        dists = dists.view(pred_bs, g.beam_size, tgt_bs)
    else:
        eval_all = lambda seqs_0, seqs_1: edit_dist_all(seqs_0, seqs_1, mode='ed')
        flat_preds = predictions.reshape(-1)
        if comp_mode == 'units':
            # NOTE(j_luo) Remove EOT's.
            flat_golds = [u[:-1] for u in units]
        elif comp_mode == 'ids':
            tgt_ids = gold_ids.cpu().numpy()
            # NOTE(j_luo) -1 for EOT's.
            tgt_lengths = lengths - 1
            flat_golds = [ids[: l] for ids, l in zip(tgt_ids, tgt_lengths)]
        elif comp_mode == 'str':
            flat_golds = forms
        dists = get_tensor(eval_all(flat_preds, flat_golds)).view(-1, g.beam_size, len(flat_golds))
    return dists


class Evaluator:

    def __init__(self,
                 model: OnePairModel,
                 dls: Dict[str, OnePairDataLoader],
                 tgt_abc: Alphabet,
                 metric_writer: Optional[MetricWriter] = None):
        """
        `dls` is a dictionary from names to dataloaders. Names are used as prefixes of metric names for the evaluator output.
        """
        self.model = model
        self.dls = dls
        self.tgt_abc = tgt_abc
        self.metric_writer = metric_writer

    def evaluate(self, stage: str, global_step: int) -> Metrics:
        self.model.eval()
        metrics = Metrics()
        with torch.no_grad():
            for name, dl in pbar(self.dls.items(), desc='eval: loader'):
                dl_metrics = self._evaluate_one_dl(stage, dl)
                metrics += dl_metrics.with_prefix(name)
        metrics = metrics.with_prefix('eval')
        logging.info(metrics.get_table(title=f'Eval: {stage}', num_paddings=8))
        if self.metric_writer is not None:
            self.metric_writer.add_metrics(metrics, global_step=global_step)
            self.metric_writer.flush()
        return metrics

    def _evaluate_one_dl(self, stage: str, dl: OnePairDataLoader) -> Metrics:
        records = list()
        K = 5
        for batch in pbar(dl, desc='eval: batch'):
            if g.eval_mode == 'edit_dist':
                batch_records = self._get_batch_records(dl, batch, K)
                records.extend(batch_records)
            else:
                scores = self.model.get_scores(batch, dl.tgt_seqs)
                top_scores, top_preds = torch.topk(scores, 5, dim='tgt_vocab')
                for pss, pis, gi in zip(top_scores, top_preds, batch.indices):
                    gold = dl.get_token_from_index(gi, 'tgt')
                    src = dl.get_token_from_index(gi, 'src')
                    record = {'source': src, 'gold_target': gold}
                    for i, (ps, pi) in enumerate(zip(pss, pis), 1):
                        pred = dl.get_token_from_index(pi, 'tgt')
                        record[f'pred_target@{i}'] = pred
                        record[f'pred_target@{i}_score'] = f'{ps:.3f}'
                    records.append(record)
        out_df = pd.DataFrame.from_records(records)
        values = ['gold_target']
        values.extend([f'pred_target@{i}' for i in range(1, K + 1)])
        values.extend([f'pred_target@{i}_score' for i in range(1, K + 1)])
        aggfunc = {'gold_target': '|'.join}
        aggfunc.update({f'pred_target@{i}': 'last' for i in range(1, K + 1)})
        aggfunc.update({f'pred_target@{i}_score': 'last' for i in range(1, K + 1)})
        if g.eval_mode == 'edit_dist':
            values.extend([f'pred_target_beam@{i}' for i in range(1, g.beam_size + 1)])
            values.extend([f'pred_target_beam@{i}_score' for i in range(1, g.beam_size + 1)])
            values.extend(['edit_dist', 'normalized_edit_dist', 'ppx'])
            aggfunc.update({f'pred_target_beam@{i}': 'last' for i in range(1, g.beam_size + 1)})
            aggfunc.update({f'pred_target_beam@{i}_score': 'last' for i in range(1, g.beam_size + 1)})
            aggfunc.update({'edit_dist': min,
                            'normalized_edit_dist': min,
                            'ppx': min})
        out_df = out_df.pivot_table(index='source', values=values,
                                    aggfunc=aggfunc)

        def is_correct(item):
            pred, gold = item
            golds = gold.split('|')
            preds = pred.split('|')
            return bool(set(golds) & set(preds))

        for i in range(1, K + 1):
            correct = out_df[[f'pred_target@{i}', 'gold_target']].apply(is_correct, axis=1)
            if i > 1:
                correct = correct | out_df[f'correct@{i - 1}']
            out_df[f'correct@{i}'] = correct
        out_folder = g.log_dir / 'predictions'
        out_folder.mkdir(exist_ok=True)
        setting = dl.setting
        out_path = str(out_folder / f'{setting.name}.{stage}.tsv')
        out_df.to_csv(out_path, sep='\t')
        logging.info(f'Predictions saved to {out_path}.')

        num_pred = len(out_df)
        metrics = Metrics()
        for i in [1, K]:
            num_correct = out_df[f'correct@{i}'].sum()
            correct = Metric(f'precision@{i}', num_correct, weight=num_pred)
            metrics += correct
        metrics += Metric('edit_dist', out_df['edit_dist'].sum(), weight=num_pred)
        metrics += Metric('normalized_edit_dist', out_df['normalized_edit_dist'].sum(), weight=num_pred)
        metrics += Metric('ppx', out_df['ppx'].sum(), weight=num_pred)
        return metrics

    def _get_batch_records(self, dl: OnePairDataLoader, batch: OnePairBatch, K: int) -> List[Dict[str, Any]]:
        hyps = self.model.predict(batch)
        # NOTE(j_luo) EOT's have been removed from translations since they don't matter in edit distance computation.
        preds, pred_lengths, _ = hyps.translate(self.tgt_abc)
        # HACK(j_luo) Pretty ugly here.
        dists = compute_edit_dist(g.comp_mode,
                                  pred_ids=hyps.tokens,
                                  lengths=dl.tgt_seqs.lengths,
                                  gold_ids=dl.tgt_seqs.ids.t(),
                                  forms=dl.tgt_vocabulary.forms,
                                  units=dl.tgt_seqs.units,
                                  predictions=preds,
                                  pred_lengths=pred_lengths,
                                  pfm=self.tgt_abc.pfm)

        weights = get_beam_probs(hyps.scores)
        w_dists = weights.align_to(..., 'tgt_vocab') * dists
        expected_dists = w_dists.sum(dim='beam')
        top_s, top_i = torch.topk(-expected_dists, K, dim='tgt_vocab')
        top_s = -top_s

        records = list()
        tgt_vocab = dl.tgt_vocabulary
        # In order to record the edit distance between the top prediction and the ground truth,
        # we need to find the index of the ground truth in the vocabulary, not in the dataset.
        tgt_ids = get_tensor([tgt_vocab.get_id_by_form(form) for form in batch.tgt_seqs.forms])
        dists_tx = Tx(dists, ['batch', 'beam', 'tgt_vocab'])
        tgt_ids_tx = Tx(tgt_ids, ['batch'])
        top_dists_tx = dists_tx.select('beam', 0)
        top_dists = top_dists_tx.each_select({'tgt_vocab': tgt_ids_tx}).data
        normalized_top_dists = top_dists.float() / (batch.tgt_seqs.lengths - 1)
        top_dists = top_dists.cpu().numpy()
        normalized_top_dists = normalized_top_dists.cpu().numpy()
        # We also report the perplexity scores.
        log_probs, _ = self.model(batch)
        ppxs = get_ce_loss(log_probs, batch, agg='batch_mean')
        ppxs = ppxs.cpu().numpy()
        for pss, pis, src, gold, pbis, pbss, top_dist, n_top_dist, ppx in zip(top_s, top_i, batch.src_seqs.forms, batch.tgt_seqs.forms, preds, weights, top_dists, normalized_top_dists, ppxs):
            record = {'source': src, 'gold_target': gold, 'edit_dist': top_dist,
                      'normalized_edit_dist': n_top_dist, 'ppx': ppx}
            for i, (pbs, pbi) in enumerate(zip(pbss, pbis), 1):
                record[f'pred_target_beam@{i}'] = pbi
                record[f'pred_target_beam@{i}_score'] = f'{pbs.item():.3f}'
            for i, (ps, pi) in enumerate(zip(pss, pis), 1):
                pred_closest = tgt_vocab[pi]['form']
                record[f'pred_target@{i}'] = pred_closest
                record[f'pred_target@{i}_score'] = f'{ps.item():.3f}'

            records.append(record)

        return records


class MctsEvaluator:

    def __init__(self, mcts: Mcts, metric_writer: Optional[MetricWriter] = None):
        self.mcts = mcts
        self.metric_writer = metric_writer

    def evaluate(self, stage: str, global_step: int) -> Metrics:
        metrics = Metrics()
        folder = g.log_dir / 'eval'
        folder.mkdir(parents=True, exist_ok=True)
        eval_tr = self.mcts.collect_episodes(self.mcts.env.start, num_episodes=1, is_eval=True)[0]
        eval_tr.save(folder / f'{stage}.path')
        logging.info(str(eval_tr))
        metrics += Metric('eval_reward', eval_tr.total_reward, 1)
        eval_tr = self.mcts.collect_episodes(self.mcts.env.start, num_episodes=1, is_eval=True, no_simulation=True)[0]
        eval_tr.save(folder / f'{stage}.path')
        logging.info(str(eval_tr))
        metrics += Metric('eval_reward_policy', eval_tr.total_reward, 1)
        metrics = metrics.with_prefix('eval')
        logging.info(metrics.get_table(title=f'Eval: {stage}', num_paddings=8))
        if self.metric_writer is not None:
            self.metric_writer.add_metrics(metrics, global_step=global_step)
            self.metric_writer.flush()
        return metrics

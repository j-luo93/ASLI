from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from dev_misc import add_argument, g, get_tensor
from dev_misc.devlib import get_array
from dev_misc.devlib.dp import EditDist
from dev_misc.devlib.tensor_x import TensorX as Tx
from dev_misc.trainlib import Metric, Metrics
from dev_misc.trainlib.tb_writer import MetricWriter
from dev_misc.utils import handle_sequence_inputs, pbar
from editdistance import eval_all
from sound_law.data.data_loader import OnePairBatch, OnePairDataLoader
from sound_law.data.dataset import EOT_ID, Alphabet
from sound_law.model.one_pair import OnePairModel

add_argument('eval_mode', dtype=str, default='prob', choices=[
             'prob', 'edit_dist'], msg='Evaluation mode using probabilities or edit distance.')


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
                # from pytorch_memlab import LineProfiler
                # with LineProfiler(self._get_batch_records) as prof:
                batch_records = self._get_batch_records(dl, batch, K)
                # print(prof.display())
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
                        record[f'pred_target@{i}_score'] = ps
                    records.append(record)
        out_df = pd.DataFrame.from_records(records)
        values = ['gold_target'] + [f'pred_target@{i}' for i in range(1, K + 1)]
        aggfunc = {'gold_target': '|'.join}
        aggfunc.update({f'pred_target@{i}': 'last' for i in range(1, K + 1)})
        if g.eval_mode == 'edit_dist':
            values += [f'pred_target_beam@{i}' for i in range(1, g.beam_size + 1)]
            aggfunc.update({f'pred_target_beam@{i}': 'last' for i in range(1, g.beam_size + 1)})
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

        num_pred = len(out_df)
        metrics = Metrics()
        for i in [1, K]:
            num_correct = out_df[f'correct@{i}'].sum()
            correct = Metric(f'precision@{i}', num_correct, weight=num_pred)
            metrics += correct
        return metrics

    def _get_batch_records(self, dl: OnePairDataLoader, batch: OnePairBatch, K: int) -> List[Dict[str, Any]]:

        @handle_sequence_inputs
        def translate(token_ids: Sequence[int]) -> Tuple[str, int]:
            ret = list()
            for tid in token_ids:
                if tid != EOT_ID:
                    # ret.append(self.tgt_abc[tid])
                    ret.append(tid)
                    # ret.append(self.tgt_abc[tid])
            # return ''.join(ret), len(ret)
            return ret, len(ret)

        hyps = self.model.predict(batch)
        pred_lengths = list()
        preds = list()
        for tokens in hyps.tokens.cpu().numpy():
            p, l = zip(*translate(tokens))
            preds.append(p)
            pred_lengths.append(l)
        preds = np.asarray(preds)
        pred_lengths = np.asarray(pred_lengths)

        # Old code.
        dists = get_tensor(eval_all(preds.reshape(-1), dl.tgt_vocab)).view(-1, g.beam_size, len(dl.tgt_vocab))
        # dists = get_tensor(eval_all(preds.reshape(-1), dl.tgt_vocab)).view(-1, g.beam_size, len(dl.tgt_vocab))

        # # Prepare tensorx.
        # pred_lengths = Tx(get_tensor(pred_lengths), ['pred_batch', 'beam'])
        # pred_tokens = Tx(hyps.tokens, ['pred_batch', 'beam', 'l'])
        # tgt_lengths = Tx(dl.tgt_seqs.lengths, ['tgt_batch'])
        # tgt_tokens = Tx(dl.tgt_seqs.ids.t(), ['tgt_batch', 'l'])

        # # Align them to the same names.
        # new_names = ['pred_batch', 'beam', 'tgt_batch']
        # pred_lengths = pred_lengths.align_to(*new_names)
        # pred_tokens = pred_tokens.align_to(*(new_names + ['l']))
        # tgt_lengths = tgt_lengths.align_to(*new_names)
        # tgt_tokens = tgt_tokens.align_to(*(new_names + ['l']))

        # # Expand them to have the same size.
        # pred_bs = pred_tokens.size('pred_batch')
        # tgt_bs = tgt_tokens.size('tgt_batch')
        # pred_lengths = pred_lengths.expand({'tgt_batch': tgt_bs})
        # pred_tokens = pred_tokens.expand({'tgt_batch': tgt_bs})
        # tgt_tokens = tgt_tokens.expand({'pred_batch': pred_bs, 'beam': g.beam_size})
        # tgt_lengths = tgt_lengths.expand({'pred_batch': pred_bs, 'beam': g.beam_size})

        # # Flatten names, preparing for DP.
        # def flatten(tx):
        #     return tx.flatten(['pred_batch', 'beam', 'tgt_batch'], 'batch')

        # pred_lengths = flatten(pred_lengths)
        # pred_tokens = flatten(pred_tokens)
        # tgt_lengths = flatten(tgt_lengths)
        # tgt_tokens = flatten(tgt_tokens)

        # dp = EditDist(pred_tokens, tgt_tokens, pred_lengths, tgt_lengths)
        # dp.run()
        # dists = dp.get_results().data
        # dists = dists.view(pred_bs, g.beam_size, tgt_bs)

        weights = hyps.scores.log_softmax(dim=-1).exp()
        w_dists = weights.align_to(..., 'tgt_vocab') * dists
        expected_dists = w_dists.sum(dim='beam')
        top_s, top_i = torch.topk(-expected_dists, K, dim='tgt_vocab')
        top_s = -top_s

        records = list()
        for pss, pis, gi, pbis, pbss in zip(top_s, top_i, batch.indices, preds, hyps.scores):
            # for scores_i, token_ids, gi in zip(scores, tokens, batch.indices):
            gold = dl.get_token_from_index(gi, 'tgt')
            src = dl.get_token_from_index(gi, 'src')
            record = {'source': src, 'gold_target': gold}
            for i, (pbs, pbi) in enumerate(zip(pbss, pbis), 1):
                record[f'pred_target_beam@{i}'] = pbi
                record[f'pred_target_beam@{i}_score'] = pbs.item()
            for i, (ps, pi) in enumerate(zip(pss, pis), 1):
                pred_closest = dl.get_token_from_index(pi, 'tgt')
                record[f'pred_target@{i}'] = pred_closest
                record[f'pred_target@{i}_score'] = ps.item()

            records.append(record)

        return records

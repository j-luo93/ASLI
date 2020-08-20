import logging
from typing import Dict, Optional

import pandas as pd
import torch

from dev_misc import g
from dev_misc.trainlib import Metric, Metrics
from dev_misc.trainlib.tb_writer import MetricWriter
from dev_misc.utils import pbar
from sound_law.data.data_loader import OnePairDataLoader
from sound_law.model.one_pair import OnePairModel


class Evaluator:

    def __init__(self,
                 model: OnePairModel,
                 dls: Dict[str, OnePairDataLoader],
                 metric_writer: Optional[MetricWriter] = None):
        """
        `dls` is a dictionary from names to dataloaders. Names are used as prefixes of metric names for the evaluator output.
        """
        self.model = model
        self.dls = dls
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

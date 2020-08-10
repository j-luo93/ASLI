from typing import Dict

import pandas as pd

from dev_misc import g
from dev_misc.trainlib import Metric, Metrics
from sound_law.data.data_loader import OnePairDataLoader
from sound_law.model.one_pair import OnePairModel


class Evaluator:

    def __init__(self, model: OnePairModel, dls: Dict[str, OnePairDataLoader]):
        """
        `dls` is a dictionary from names to dataloaders. Names are used as prefixes of metric names for the evaluator output.
        """
        self.model = model
        self.dls = dls

    def evaluate(self, stage: str) -> Metrics:
        metrics = Metrics()
        for name, dl in self.dls.items():
            dl_metrics = self._evaluate_one_dl(stage, dl)
            metrics += dl_metrics.with_prefix_(name)
        return metrics

    def _evaluate_one_dl(self, stage: str, dl: OnePairDataLoader) -> Metrics:
        records = list()
        for batch in dl:
            scores = self.model.get_scores(batch, dl.tgt_seqs)
            preds = scores.max(dim='tgt_vocab')[1]
            assert preds.shape == batch.indices.shape
            for pi, gi in zip(preds, batch.indices):
                pred = dl.get_token_from_index(pi, 'tgt')
                gold = dl.get_token_from_index(gi, 'tgt')
                src = dl.get_token_from_index(gi, 'src')
                records.append({'source': src, 'gold_target': gold, 'pred_target': pred})
        out_df = pd.DataFrame.from_records(records)
        out_df = out_df.pivot_table(index='source', values=['gold_target', 'pred_target'],
                                    aggfunc={'gold_target': '|'.join,
                                             'pred_target': '|'.join}
                                    )

        def is_correct(item):
            pred, gold = item
            golds = gold.split('|')
            preds = pred.split('|')
            return bool(set(golds) & set(preds))

        out_df['correct'] = out_df[['pred_target', 'gold_target']].apply(is_correct, axis=1)
        out_folder = g.log_dir / 'predictions'
        out_folder.mkdir(exist_ok=True)
        setting = dl.setting
        out_path = str(out_folder / f'{setting.name}.{stage}.tsv')
        out_df.to_csv(out_path, sep='\t')

        num_pred = len(out_df)
        num_correct = out_df['correct'].sum()
        correct = Metric('correct', num_correct, weight=num_pred)
        return Metrics(correct)

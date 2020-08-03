from dev_misc.trainlib import Metric, Metrics
from dev_misc import g
import pandas as pd
from sound_law.data.data_loader import OnePairDataLoader
from sound_law.model.one_pair import OnePairModel


class OnePairEvaluator:

    def __init__(self, model: OnePairModel, dl: OnePairDataLoader):
        self.model = model
        self.dl = dl

    def evaluate(self, stage: str) -> Metrics:
        num_pred = 0
        num_correct = 0
        records = list()
        for batch in self.dl:
            scores = self.model.get_scores(batch, self.dl.tgt_seqs)
            preds = scores.max(dim='tgt_vocab')[1]
            assert preds.shape == batch.indices.shape
            num_pred += len(batch)
            correct = (batch.indices == preds)
            num_correct += correct.sum()
            for pi, gi, corr in zip(preds, batch.indices, correct.cpu().numpy()):
                pred = self.dl.get_token_from_index(pi, 'tgt')
                gold = self.dl.get_token_from_index(gi, 'tgt')
                src = self.dl.get_token_from_index(gi, 'src')
                records.append({'source': src, 'gold_target': gold, 'pred_target': pred, 'correct': corr})
        out_df = pd.DataFrame.from_records(records)
        out_folder = g.log_dir / 'predictions'
        out_folder.mkdir(exist_ok=True)
        out_path = str(out_folder / f'{stage}.tsv')
        out_df.to_csv(out_path, sep='\t', index=None)

        correct = Metric('correct', num_correct, weight=num_pred)
        return Metrics(correct)

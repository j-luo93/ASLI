from dev_misc.trainlib import Metric, Metrics
from sound_law.data.data_loader import OnePairDataLoader
from sound_law.model.one_pair import OnePairModel


class OnePairEvaluator:

    def __init__(self, model: OnePairModel, dl: OnePairDataLoader):
        self.model = model
        self.dl = dl

    def evaluate(self, *args, **kwargs) -> Metrics:  # TODO(j_luo) Fix the args
        num_pred = 0
        num_correct = 0
        for batch in self.dl:
            scores = self.model.get_scores(batch, self.dl.tgt_seqs)
            preds = scores.max(dim='tgt_vocab')[1]
            assert preds.shape == batch.indices.shape
            num_pred += len(batch)
            num_correct += (batch.indices == preds).sum()
        correct = Metric('correct', num_correct, weight=num_pred)
        return Metrics(correct)

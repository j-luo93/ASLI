import torch
import torch.nn as nn

from dev_misc import FT, g
from sound_law.data.data_loader import OnePairBatch
from sound_law.data.dataset import SOT_ID

from .one_pair import OnePairModel


class OneToManyModel(OnePairModel):

    def __init__(self, num_src_chars: int, num_tgt_chars: int, num_tgt_langs: int):
        super().__init__(num_src_chars, num_tgt_chars)
        self.lang_emb = nn.Embedding(num_tgt_langs, g.char_emb_size)

    def _get_log_probs(self, batch: OnePairBatch, use_target: bool = True, max_length: int = None) -> FT:
        src_emb, output, state = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        lang_emb = self.lang_emb.weight[batch.tgt_seqs.lang_id]
        target = batch.tgt_seqs.ids if use_target else None
        log_probs = self.decoder(SOT_ID, src_emb,
                                 output, batch.src_seqs.paddings,
                                 target=target,
                                 max_length=max_length,
                                 lang_emb=lang_emb)
        return log_probs

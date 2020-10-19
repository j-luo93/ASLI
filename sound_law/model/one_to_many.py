from typing import Optional, Sequence, Tuple, Dict

import torch
import torch.nn as nn

from dev_misc import FT, LT, add_argument, g
from sound_law.data.data_loader import OnePairBatch
from sound_law.data.dataset import SOT_ID

from .base_model import BaseModel
from .module import LanguageEmbedding


class OneToManyModel(BaseModel):

    add_argument('lang_emb_mode', default='mean', dtype=str,
                 choices=['random', 'mean', 'lang2vec', 'wals'], msg='Mode for the language embedding module.')
    add_argument('l2v_feature_set', default=None, dtype=str,
                 choices=['phonology_average', 'phonology_wals', 'phonology_ethnologue', 'learned'], msg='Which feature set to use for the lang2vec language embeddings.')

    def __init__(self, num_src_chars: int, num_tgt_chars: int, num_tgt_langs: int, unseen_idx: int,
                 lang2id: Optional[Dict[str, int]] = None,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):
        super().__init__(num_src_chars, num_tgt_chars,
                         phono_feat_mat=phono_feat_mat, special_ids=special_ids)
        self.lang_emb = LanguageEmbedding(num_tgt_langs, g.char_emb_size,
                                          unseen_idx=unseen_idx,
                                          lang2id=lang2id,
                                          mode=g.lang_emb_mode,
                                          dropout=g.dropout)

    def _prepare_lang_emb(self, batch: OnePairBatch) -> FT:
        return self.lang_emb(batch.tgt_seqs.lang_id)

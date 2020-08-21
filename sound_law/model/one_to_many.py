from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dev_misc import FT, LT, add_argument, g
from sound_law.data.data_loader import OnePairBatch
from sound_law.data.dataset import SOT_ID

from .module import LanguageEmbedding
from .one_pair import OnePairModel, CnnEncoderOnePairModel


class OneToManyModel(OnePairModel):

    add_argument('lang_emb_mode', default='mean', dtype=str,
                 choices=['random', 'mean'], msg='Mode for the language embedding module.')

    def __init__(self, num_src_chars: int, num_tgt_chars: int, num_tgt_langs: int, unseen_idx: int,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):
        super().__init__(num_src_chars, num_tgt_chars,
                         phono_feat_mat=phono_feat_mat, special_ids=special_ids)
        self.lang_emb = nn.Embedding(num_tgt_langs, g.char_emb_size)
        self.lang_emb = LanguageEmbedding(num_tgt_langs, g.char_emb_size,
                                          unseen_idx=unseen_idx, mode=g.lang_emb_mode)

    def forward(self, batch: OnePairBatch, use_target: bool = True, max_length: int = None) -> Tuple[FT, FT]:
        src_emb, output, state = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        lang_emb = self.lang_emb(batch.tgt_seqs.lang_id)
        target = batch.tgt_seqs.ids if use_target else None
        log_probs, almt_distrs = self.decoder(SOT_ID, src_emb,
                                              output, batch.src_seqs.paddings,
                                              target=target,
                                              max_length=max_length,
                                              lang_emb=lang_emb)
        return log_probs, almt_distrs


class CnnEncoderOneToManyModel(CnnEncoderOnePairModel):
    
    def __init__(self, num_src_chars: int, num_tgt_chars: int, num_tgt_langs: int, unseen_idx: int,
                 lang2id: Optional[Dict[str, int]] = None,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):
        super().__init__(num_src_chars, num_tgt_chars, lang2id=lang2id,
                         phono_feat_mat=phono_feat_mat, special_ids=special_ids)
        self.lang_emb = LanguageEmbedding(num_tgt_langs, g.char_emb_size,
                                          unseen_idx=unseen_idx,
                                          lang2id=lang2id,
                                          mode=g.lang_emb_mode)
    
    def forward(self, batch: OnePairBatch, use_target: bool = True, max_length: int = None) -> Tuple[FT, FT]:
        src_emb, output, state = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        lang_emb = self.lang_emb(batch.tgt_seqs.lang_id)
        target = batch.tgt_seqs.ids if use_target else None
        log_probs, almt_distrs = self.decoder(SOT_ID, src_emb,
                                              output, batch.src_seqs.paddings,
                                              target=target,
                                              max_length=max_length,
                                              lang_emb=lang_emb)
        return log_probs, almt_distrs


class CnnEncoderOneToManyModel(CnnEncoderOnePairModel):
    
    def __init__(self, num_src_chars: int, num_tgt_chars: int, num_tgt_langs: int, unseen_idx: int,
                 lang2id: Optional[Dict[str, int]] = None,
                 phono_feat_mat: Optional[LT] = None,
                 special_ids: Optional[Sequence[int]] = None):
        super().__init__(num_src_chars, num_tgt_chars, lang2id=lang2id,
                         phono_feat_mat=phono_feat_mat, special_ids=special_ids)
        self.lang_emb = LanguageEmbedding(num_tgt_langs, g.char_emb_size,
                                          unseen_idx=unseen_idx,
                                          lang2id=lang2id,
                                          mode=g.lang_emb_mode,
                                          dropout=g.dropout)
    
    def forward(self, batch: OnePairBatch, use_target: bool = True, max_length: int = None) -> Tuple[FT, FT]:
        src_emb, output, state = self.encoder(batch.src_seqs.ids, batch.src_seqs.lengths)
        lang_emb = self.lang_emb(batch.tgt_seqs.lang_id)
        target = batch.tgt_seqs.ids if use_target else None
        log_probs, almt_distrs = self.decoder(SOT_ID, src_emb,
                                              output, batch.src_seqs.paddings,
                                              target=target,
                                              max_length=max_length,
                                              lang_emb=lang_emb)
        return log_probs, almt_distrs


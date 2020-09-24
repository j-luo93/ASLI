from argparse import ArgumentParser

import pandas as pd
import torch

from dev_misc import g, get_tensor
from sound_law.main import setup
from sound_law.model.base_model import get_emb_params
from sound_law.model.module import PhonoEmbedding
from sound_law.train.manager import OneToManyManager

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('saved_g_path', type=str, help='Path to the saved g.')
    parser.add_argument('saved_model_path', type=str, help='Path to the saved model.')
    parser.add_argument('out_name', type=str, help='Path to save the output. No suffix should be included.')
    args = parser.parse_args()

    if '.' in args.out_name:
        raise ValueError(f'No suffix should be included.')

    initiator = setup()
    initiator.run(saved_g_path=args.saved_g_path)
    _, _, abc, _ = OneToManyManager.prepare_raw_data()
    assert g.share_src_tgt_abc

    sd = torch.load(args.saved_model_path)

    emb_params = get_emb_params(len(abc),
                                phono_feat_mat=get_tensor(abc.pfm),
                                special_ids=get_tensor(abc.special_ids))
    emb = PhonoEmbedding.from_params(emb_params)
    prefix = 'encoder.embedding'
    emb.load_state_dict({'weight': sd[f'{prefix}.weight'],
                         'special_weight': sd[f'{prefix}.special_weight'],
                         'special_mask': sd[f'{prefix}.special_mask'],
                         'pfm': sd[f'{prefix}.pfm']})
    emb.cuda()

    char_emb = emb.char_embedding.detach().cpu().numpy()
    size = char_emb.shape[-1]
    cols = [f'vec_{i}' for i in range(size)]
    df = pd.DataFrame(char_emb, columns=cols)
    df.to_csv(args.out_name + '.tsv', sep='\t', index=None, header=None)

    meta_df = pd.DataFrame(list(abc), columns=['unit'])
    meta_df.to_csv(args.out_name + '.meta.tsv', sep='\t', index=None, header=None)

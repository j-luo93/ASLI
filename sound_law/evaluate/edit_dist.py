from typing import Sequence, Tuple, Union, overload

import numpy as np
from lingpy.align.pairwise import pw_align

from dev_misc import g
from dev_misc.devlib import NDA
from dev_misc.utils import handle_sequence_inputs, pbar
from editdistance import eval as ed_eval
from editdistance import eval_all as ed_eval_all
from editdistance import eval_batch as ed_eval_batch
from sound_law.data.alphabet import EOT_ID, Alphabet

Number = Union[int, float]


def translate(token_ids: Sequence[int], abc: Alphabet) -> Tuple[str, int]:
    ret = list()
    for tid in token_ids:
        if tid != EOT_ID:
            if g.comp_mode in ['units', 'str', 'ids_gpu']:
                ret.append(abc[tid])
            else:
                ret.append(tid)
    if g.comp_mode in ['units', 'ids']:
        return ret, len(ret)
    else:
        return ''.join(ret), len(ret)


def edit_dist(seq_0: str, seq_1: str, mode: str) -> Number:
    """A master function for dispatching different methods of computing edit distance."""
    if mode == 'ed':
        return ed_eval(seq_0, seq_1)
    elif mode == 'global':
        l0 = len(seq_0)
        l1 = len(seq_1)
        return max(l0, l1) - pw_align(seq_0, seq_1, mode='global')[-1]
    else:
        raise ValueError(f'Unrecognized value "{mode}" for mode.')


def edit_dist_all(seqs_0: Sequence[str], seqs_1: Sequence[str], mode: str) -> NDA:
    if mode == 'ed':
        return ed_eval_all(seqs_0, seqs_1, g.num_threads)

    ret = list()
    for seq_0 in seqs_0:
        ret.append([edit_dist(seq_0, seq_1, mode) for seq_1 in seqs_1])
    return np.asarray(ret)


def edit_dist_batch(seqs_0: Sequence[str], seqs_1: Sequence[str], mode: str) -> NDA:
    if len(seqs_0) != len(seqs_1):
        raise ValueError(f'Expect equal values, but got {len(seqs_0)} and {len(seqs_1)}.')
    if mode == 'ed':
        return ed_eval_batch(seqs_0, seqs_1, g.num_threads)

    ret = [edit_dist(seq_0, seq_1, mode) for seq_0, seq_1 in zip(seqs_0, seqs_1)]
    return np.asarray(ret)

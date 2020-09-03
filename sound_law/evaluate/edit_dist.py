from typing import Sequence, Union, overload

import numpy as np
from lingpy.align.pairwise import pw_align

from dev_misc.devlib import NDA
from editdistance import eval as ed_eval
from editdistance import eval_all as ed_eval_all

Number = Union[int, float]


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
        return ed_eval_all(seqs_0, seqs_1)

    ret = list()
    for seq_0 in seqs_0:
        ret.append([edit_dist(seq_0, seq_1, mode) for seq_1 in seqs_1])
    return np.asarray(ret)

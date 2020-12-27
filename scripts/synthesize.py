from __future__ import annotations

import pickle
import sys
from argparse import ArgumentParser
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from dev_misc import g
from dev_misc.arglib import disable_duplicate_check, set_argument
from dev_misc.trainlib.base_trainer import BaseTrainer
from sound_law.data.alphabet import ANY_ID, EMP_ID, EOT_ID, SOT_ID
from sound_law.data.cognate import CognateRegistry
from sound_law.main import setup
from sound_law.rl.action import SoundChangeActionSpace
from sound_law.rl.env import SoundChangeEnv
from sound_law.rl.mcts.mcts_fast import (PyActionSpace, PyEnv, PySiteSpace,
                                         PyWordSpace)
from sound_law.rl.trajectory import VocabState
from sound_law.train.manager import OnePairManager

dispatch = dict()


def register(cls):
    name = cls.__name__
    assert name not in dispatch
    obj = cls()
    dispatch[name] = obj
    return cls


@register
class Fronting:

    def form_change(self, x) -> str:
        return x.replace('a', 'ae').replace('o', 'oe').replace('u', 'y')

    def ipa_change(self, x) -> str:
        return x.replace('ɑ', 'æ').replace('o', 'ø').replace('u', 'y')


@register
class E2I:

    def form_change(self, x) -> str:
        return x.replace('e', 'i')

    def ipa_change(self, x) -> str:
        return x.replace('e', 'i')


def get_unit_seqs(vocab, abc):
    unit_seqs = list()
    for id_seq in vocab:
        unit_seq = [abc[i] for i in id_seq if i not in abc.special_ids]
        unit_seqs.append(unit_seq)
    return unit_seqs


def get_all_chars(state, abc):
    all_chars = set()
    for unit_seq in get_unit_seqs(state.vocab, abc):
        all_chars.update(unit_seq)
    return all_chars


def get_units(state, abc):
    ret = list()
    for unit_seq in get_unit_seqs(state.vocab, abc):
        ret.append(' '.join(unit_seq))
    return ret


def get_record(ipas):
    return {
        'transcription': ''.join(ipas),
        'ipa': ''.join(ipas),
        'tokens': ' '.join(ipas),
        'split': 'train'
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('src_path', type=str, help='Path to the src tsv file.')
    parser.add_argument('out_path', type=str, help='Output path.')
    parser.add_argument('mode', type=str, help='Configuration name.')
    parser.add_argument('--length', type=int, help='Length of synthesizing random rules.')
    parser.add_argument('--random_seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--options', default='', type=str, help='Extra options')
    args = parser.parse_args()

    if args.mode == 'random':

        sys.argv = f'''
        sound_law/main.py
            --config OPRLPgmcGot
            --mcts_config LargeSims
            --no_use_value_guidance
            --use_conditional
            {args.options}
        '''.split()

        initiator = setup()
        initiator.run()

        set_argument('data_path', 'data/wikt', _force=True)
        set_argument('segments_dump_path', 'data/nel_segs.pkl', _force=True)
        set_argument('ngram_path', 'data/nel_ngram.pkl', _force=True)

        manager = OnePairManager()
        dl = manager.dl_reg.get_loaders_by_name('rl')
        src_seqs = dl.entire_batch.src_seqs
        s_arr = np.ascontiguousarray(src_seqs.ids.t().cpu().numpy())
        s_lengths = np.ascontiguousarray(src_seqs.lengths.t().cpu().numpy())
        tgt_seqs = dl.entire_batch.tgt_seqs
        t_arr = np.ascontiguousarray(tgt_seqs.ids.t().cpu().numpy())
        t_lengths = np.ascontiguousarray(tgt_seqs.lengths.t().cpu().numpy())
        py_ss = PySiteSpace(SOT_ID, EOT_ID, ANY_ID, EMP_ID)
        py_ws = PyWordSpace(py_ss, manager.tgt_abc.dist_mat, 2.0, t_arr, t_lengths)
        action_space = SoundChangeActionSpace(py_ss, py_ws, g.dist_threshold,
                                              g.site_threshold, g.num_workers, manager.tgt_abc)
        env = SoundChangeEnv(py_ws, action_space, s_arr, s_lengths, g.final_reward, g.step_penalty)

        init_n_chars = len(get_all_chars(env.start, manager.tgt_abc))
        print(init_n_chars)
        state = env.start
        print(get_units(state, manager.tgt_abc))
        print(f'Distance: {state.dist:.3f}')
        path = [state]
        np.random.seed(args.random_seed)
        for i in range(args.length):
            while True:
                env.action_space.set_action_allowed(state)
                best_i = np.random.choice(len(state.action_allowed))
                print(len(state.action_allowed), 'allowed.')
                # for i, a in enumerate(state.action_allowed):
                #     new, _ = env.step(state, i, a)
                #     from sound_law.rl.mcts.mcts_fast import PyStop
                #     print(env.action_space.get_action(a))
                # 1 / 0
                #assert new.dist < state.dist or PyStop == a, new.dist
                action_id = state.action_allowed[best_i]
                action = env.action_space.get_action(action_id)
                next_state, reward = env.step(state, best_i, action_id)
                if abs(len(get_all_chars(next_state, manager.tgt_abc)) - init_n_chars) > 3:
                    print('Too many characters have changed, retrying...')
                    continue
                print(f'step {i + 1} finished.')
                state = next_state
                print('-' * 20)
                print(env.action_space.get_action(state.prev_action[1]))
                print(get_units(state, manager.tgt_abc))
                print(f'Distance: {state.dist:.3f}')
                path.append(state)
                break

        records = list()
        for unit_seq in get_unit_seqs(path[-1].vocab, manager.tgt_abc):
            records.append(get_record(unit_seq))
        df = pd.DataFrame(records)
        df.to_csv(args.out_path, sep='\t', index=None)
    else:
        converter = dispatch[args.mode]

        df = pd.read_csv(args.src_path, sep='\t', keep_default_na=True, error_bad_lines=False)
        out = df.copy()
        with open(args.out_path, 'w') as fout:
            out['transcription'] = out['transcription'].apply(converter.form_change)
            out['ipa'] = out['ipa'].apply(converter.ipa_change)
            out['tokens'] = out['tokens'].apply(converter.ipa_change)
        out.to_csv(args.out_path, sep='\t', index=None)

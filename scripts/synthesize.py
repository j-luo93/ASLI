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
from sound_law.data.cognate import CognateRegistry
from sound_law.main import setup
from sound_law.rl.action import SoundChangeActionSpace
from sound_law.rl.env import SoundChangeEnv
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
    args = parser.parse_args()

    if args.mode == 'random':

        sys.argv = 'sound_law/main.py --config OPRLFakeR5 --mcts_config MoreSims --no_use_value_guidance --use_conditional'.split()

        initiator = setup()
        initiator.run()

        set_argument('data_path', 'data/wikt', _force=True)
        set_argument('phoible_path', 'data/phoible_segs.pkl', _force=True)

        manager = OnePairManager()
        dl = manager.dl_reg.get_loaders_by_name('rl')
        env = SoundChangeEnv(dl.init_state, dl.end_state, manager.action_space, g.final_reward, g.step_penalty)

        init_n_chars = len(get_all_chars(dl.init_state, manager.tgt_abc))
        print(init_n_chars)
        state = dl.init_state
        path = [state]
        for i in range(args.length):
            while True:
                best_i = np.random.choice(len(state.action_allowed))
                print(len(state.action_allowed), 'allowed.')
                action_id = state.action_allowed[best_i]
                action = manager.action_space.get_action(action_id)
                next_state, done, reward = env.step(state, best_i, action)
                if abs(len(get_all_chars(next_state, manager.tgt_abc)) - init_n_chars) > 3:
                    print('Too many characters have changed, retrying...')
                    continue
                print(f'step {i + 1} finished.')
                state = next_state
                path.append(state)
                break

        for i, state in enumerate(path):
            if i > 0:
                print(manager.action_space.get_action(state.prev_action[1]))
            print(get_units(state, manager.tgt_abc))
            print('-' * 20)

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

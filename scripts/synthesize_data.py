import random
import sys
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm

from sound_law.data.alphabet import EMP_ID, NULL_ID, Alphabet
from sound_law.main import OnePairManager, setup
from sound_law.rl.action import SoundChangeAction
from sound_law.rl.env import SoundChangeEnv
from sound_law.rl.mcts_cpp import PyPS_SAMPLE_AC
from sound_law.rl.rule import HandwrittenRule
from sound_law.rl.trajectory import VocabState, int2st
from sound_law.utils import run_with_argument


# @st.cache(hash_funcs={OnePairManager: id})
def load_manager() -> OnePairManager:
    initiator = setup()
    initiator.run()
    return OnePairManager()


def get_action_from_path(path: List[int]) -> SoundChangeAction:
    before, rtype, after, pre, d_pre, post, d_post = path
    rtype = int2st[rtype]
    return SoundChangeAction(before, after, rtype, pre, d_pre, post, d_post)


def is_merger(action: SoundChangeAction, state: VocabState, abc: Alphabet) -> bool:
    """Whether this `action` is a merger."""
    after_unit = abc[abc.unit2base[action.after_id]]
    return after_unit in state.alphabet


def get_num_affected(env: SoundChangeEnv, state: VocabState, action: SoundChangeAction) -> int:
    if action.before_id == NULL_ID:
        return 0
    else:
        return env.get_num_affected(state, action)


def is_split(action: SoundChangeAction, state: VocabState, abc: Alphabet, env: SoundChangeEnv) -> Tuple[bool, int]:
    """Whether this `action` is a split."""
    num_occ = state.get_num_occurences(abc[action.before_id])
    num_aff = get_num_affected(env, state, action)
    return num_aff > 0 and num_aff < num_occ, num_aff


def is_loss(action: SoundChangeAction) -> bool:
    """Whether this `action` is a loss."""
    return action.after_id == EMP_ID


@dataclass
class ActionInfo:
    action: SoundChangeAction
    is_merger_bool: bool
    is_split_bool: bool
    is_loss_bool: bool
    num_aff: int

# @st.cache(suppress_st_warning=True, hash_funcs={OnePairManager: id, VocabState: id, SoundChangeEnv: id})


def generate_episodes(manager: OnePairManager, num_episodes: int):

    def sample_action(state: VocabState, step: int, played_path):
        """Sample one action that is not STOP."""
        manager.env.evaluate(state,
                             np.zeros([6, len(manager.tgt_abc)], dtype='float32'),
                             np.zeros([len(manager.tgt_abc)], dtype='float32'))
        while True:
            new_state = manager.mcts.select_one_random_step(state)
            new_path = manager.mcts.play(state, step, PyPS_SAMPLE_AC, 1.0)
            action = get_action_from_path(new_path.get_last_action_vec())
            manager.env.clear_stats(state, True)
            num_aff = get_num_affected(manager.env, state, action)
            if action.before_id != NULL_ID and num_aff >= 2:
                if played_path is None:
                    played_path = new_path
                else:
                    played_path.merge(new_path)
                return action, played_path

    episode_status = st.text('Episode:')
    episode_pbar = st.progress(0.0)
    step_status = st.text('Step:')
    step_pbar = st.progress(0)
    config_df_records = list()
    states = list()
    action_seqs = list()
    for episode in range(num_episodes):
        root = manager.env.start
        step_pbar.progress(0)
        n_merger = n_split = n_loss = 0
        played_path = None
        action_seq = list()
        # tmp_states = list()
        # print('==============================')
        for step in range(10):
            # print(f'Step {step}')
            action, played_path = sample_action(root, step, played_path)
            is_merger_bool = is_merger(action, root, manager.tgt_abc)
            is_split_bool, num_aff = is_split(action, root, manager.tgt_abc, manager.env)
            is_loss_bool = is_loss(action)
            n_merger += is_merger_bool
            n_split += is_split_bool
            n_loss += is_loss_bool
            action_seq.append(ActionInfo(action, is_merger_bool, is_split_bool, is_loss_bool, num_aff))
            root = manager.env.apply_action(root, action)
            # tmp_states.append(root)
            step_pbar.progress((step + 1) * 10)
            step_status.text(f'Step: {step + 1} / 10')
        states.append(root)
        action_seqs.append(action_seq)
        episode_pbar.progress((episode + 1) / num_episodes)
        episode_status.text(f'Episode: {episode + 1} / {num_episodes}')
        config_df_records.append({'n_merger': n_merger, 'n_split': n_split, 'n_loss': n_loss})
    return pd.DataFrame(config_df_records), states, action_seqs


if __name__ == "__main__":
    parser = ArgumentParser()
    src_path = 'data/wikt/pgmc-got/pgmc.tsv'
    sys.argv = 'dummy.py --config OPRLPgmcGot --mcts_config SmallSims --site_threshold 1 --dist_threshold 1000.0'.split()
    src_df = pd.read_csv(src_path, sep='\t')
    manager = load_manager()
    config_df, states, action_seqs = generate_episodes(manager, 100)
    st.write(config_df)
    st.subheader('n_merger')
    st.bar_chart(config_df['n_merger'].value_counts())
    st.subheader('n_split')
    st.bar_chart(config_df['n_split'].value_counts())
    st.subheader('n_loss')
    st.bar_chart(config_df['n_loss'].value_counts())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(
        config_df["n_merger"] + np.random.random(100) * 0.3 - 0.15,
        config_df["n_split"] + np.random.random(100) * 0.3 - 0.15,
    )

    ax.set_xlabel("n_merger")
    ax.set_ylabel("n_split")

    st.write(fig)

    wikt_folder = Path('data/wikt')
    for i, (state, action_seq) in enumerate(zip(states, action_seqs), 1):
        tgt_df = pd.DataFrame()
        tgt_df['transcription'] = state.word_list
        tgt_df['ipa'] = state.word_list
        tgt_df['tokens'] = [' '.join(segs[1:-1]) for segs in state.segment_list]
        for col in ['transcription', 'ipa', 'tokens']:
            tgt_df[col] = tgt_df[col].apply(lambda x: x.replace('{+}', '').replace('{-}', ''))
        tgt_df['split'] = 'train'
        out_folder = wikt_folder / f'pgmc-rand{i}'
        out_folder.mkdir(parents=True, exist_ok=True)
        src_df.to_csv(str(out_folder / 'pgmc.tsv'), sep='\t', index=None)
        tgt_df.to_csv(str(out_folder / f'rand{i}.tsv'), sep='\t', index=None)

        action_records = list()
        for action_info in action_seq:
            record = asdict(action_info)
            record['action'] = str(action_info.action)
            action_records.append(record)
        action_df = pd.DataFrame(action_records)

        action_df.to_csv(str(out_folder / 'action_seq.tsv'), sep='\t', index=None)

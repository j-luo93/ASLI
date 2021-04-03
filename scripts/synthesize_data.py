import sys
from argparse import ArgumentParser

import pandas as pd
import streamlit as st

from sound_law.main import OnePairManager, setup
from sound_law.utils import run_with_argument


@st.cache(hash_funcs={OnePairManager: id})
def load_manager() -> OnePairManager:
    initiator = setup()
    initiator.run()
    return OnePairManager()


if __name__ == "__main__":
    parser = ArgumentParser()
    src_path = 'data/wikt/pgmc-got/pgmc.tsv'
    sys.argv = 'dummy.py --config OPRLPgmcGot --mcts_config SmallSims --site_threshold 1 --dist_threshold 1000.0'.split()
    src_df = pd.read_csv(src_path, sep='\t')
    manager = load_manager()
    results = manager.env.expand_all_actions(manager.env.start)
    results[0]
    src_df

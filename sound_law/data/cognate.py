import logging
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, NewType, Set, Tuple

import pandas as pd

from dev_misc import Arg, add_argument, add_check, g
from dev_misc.utils import handle_sequence_inputs

from .alphabet import Alphabet
from .dataset import OnePairDataset
from .setting import Setting

Lang = NewType('Lang', str)
DF = pd.DataFrame

add_argument('use_stress', dtype=bool, default=True, msg='Flag to use stress.')
add_argument('use_duration', dtype=bool, default=True, msg='Flag to use duration (long or short).')
add_argument('use_diacritics', dtype=bool, default=True, msg='Flag to use diacritics.')
add_argument('use_duplicate_phono', dtype=bool, default=True,
             msg='Whether to keep duplicate symbols based on their phonological features.')
add_argument('noise_level', dtype=float, default=0.0, msg='Noise level on the target side.')
add_check(
    (Arg('use_duplicate_phono') == False) | (Arg('separate_output') == True) | (Arg('use_phono_features') == False))


@handle_sequence_inputs
def _preprocess(s: str) -> str:
    s = unicodedata.normalize('NFD', s)

    def one_pass(s):
        if not g.use_stress and s[0] == "ˈ":
            s = s[1:]
        if not g.use_duration and s[-1] == 'ː':
            s = s[:-1]
        if not g.use_diacritics and unicodedata.category(s[-1]) in ['Mn', 'Lm']:
            s = s[:-1]
        return s

    while True:
        new_s = one_pass(s)
        if s == new_s:
            break
        s = new_s
    return s


def get_paths(data_path: Path, src_lang: Lang, tgt_lang: Lang) -> Tuple[Path, Path]:
    prefix = data_path / f'{src_lang}-{tgt_lang}'
    src_path = f'{prefix / src_lang}.tsv'
    tgt_path = f'{prefix / tgt_lang}.tsv'
    return Path(src_path).resolve(), Path(tgt_path).resolve()


def postprocess(pre_unit_seq: List[str], std_func: Callable[[str], str], abc: Alphabet) -> dict:
    """Postprocess the unit sequence by applying a standardization function and indexing every unit."""
    if g.use_duplicate_phono:
        post_unit_seq = pre_unit_seq
    else:
        post_unit_seq = std_func(pre_unit_seq)
    id_seq = [abc[u] for u in post_unit_seq]
    form = ''.join(post_unit_seq)
    ret = {
        'post_unit_seq': post_unit_seq,
        'id_seq': id_seq,
        'form': form
    }
    return ret


class CognateRegistry:
    """A registry to hold all cognate information for many languages."""

    def __init__(self):
        # This maps each language to a dictionary that maps a path to a data frame.
        self._lang2dfs: Dict[Lang, Dict[Path, DF]] = defaultdict(dict)

        # A dictionary from language to alphabet. After its alphabet
        # has been prepared, no more files can be added for that language.
        self._lang2abc: Dict[Lang, Alphabet] = dict()

        # A dictionary that maps a language pair to a pair of data frames.
        self._pair2dfs: Dict[Tuple[Lang, Lang], Tuple[DF, DF]] = dict()

    def add_pair(self, data_path: Path, src_lang: Lang, tgt_lang: Lang) -> Tuple[DF, DF]:
        """Add a pair of languages."""
        if (src_lang, tgt_lang) in self._pair2dfs:
            raise RuntimeError(f'Language pair {src_lang}-{tgt_lang} has already been added.')
        src_path, tgt_path = get_paths(data_path, src_lang, tgt_lang)
        src_df = self.add_file(src_lang, src_path)
        tgt_df = self.add_file(tgt_lang, tgt_path)
        self._pair2dfs[(src_lang, tgt_lang)] = (src_df, tgt_df)
        return src_df, tgt_df

    def add_file(self, lang: Lang, path: Path) -> DF:
        """Read the file at `path` for language `lang`."""
        if lang in self._lang2abc:
            raise RuntimeError(
                f'An alphabet has already been prepared for language {lang}. No more files can be added.')
        # Always use the resolved path as the key.
        path = path.resolve()
        # Skip this process if this file has already been added.
        if lang in self._lang2dfs and path in self._lang2dfs[lang]:
            logging.warn(f'File at {path} has already been added for language {lang}.')
        # Actually add this file.
        else:
            df = pd.read_csv(str(path), sep='\t', keep_default_na=True)
            df['cognate_id'] = range(len(df))
            if g.input_format == 'wikt':
                df = df.copy()
                df['tokens'] = df['tokens'].str.split('|')
                df = df.explode('tokens')
            # In-place to add a unit_seq column to store preprocessed data.
            col = 'tokens' if g.input_format == 'wikt' else 'parsed_tokens'  # Use parsed tokens if possible.
            df['pre_unit_seq'] = df[col].str.split().apply(_preprocess)
            # NOTE(j_luo) Add noise to the target tokens by randomly duplicating one character.
            if g.noise_level > 0.0:
                logging.imp(f'Adding noise to the target tokens with level {g.noise_level}.')
                random.seed(g.random_seed)

                def add_noise(token):
                    if random.random() < g.noise_level:
                        pos = random.randint(0, len(token) - 1)
                        token = token[:pos] + [token[pos]] + token[pos:]
                    return token

                df['pre_unit_seq'] = df['pre_unit_seq'].apply(add_noise)

            df = df.set_index('cognate_id')
            self._lang2dfs[lang][path] = df
            logging.info(f'File at {path} has been added for language {lang}.')
        return self._lang2dfs[lang][path]

    def get_alphabet(self, lang: Lang) -> Alphabet:
        return self._lang2abc[lang]

    def prepare_alphabet(self, *langs: Lang) -> Alphabet:
        if any(lang in self._lang2abc for lang in langs):
            raise TypeError(f'Some lang in {langs} has been prepared.')
        # Get all relevant data frames.
        dfs = list()
        for lang in langs:
            for source, df in self._lang2dfs[lang].items():
                df['source'] = source
                dfs.append(df)
        df = pd.concat(dfs)

        # Get and register the alphabet.
        contents = df['pre_unit_seq']
        sources = df['source'].tolist()
        abc = Alphabet(','.join(langs), contents, sources=sources)
        for lang in langs:
            self._lang2abc[lang] = abc

        # Post-progress the unit seqs if needed.
        std_func = handle_sequence_inputs(lambda s: abc.standardize(s))
        cols = ['post_unit_seq', 'id_seq', 'form']  # These are the columns to add.
        for lang in langs:
            for df in self._lang2dfs[lang].values():
                records = df['pre_unit_seq'].apply(postprocess, std_func=std_func, abc=abc).tolist()
                # NOTE(j_luo) Make sure to use the same index as the original `df` since duplicate indices indicate multiple references.
                post = pd.DataFrame(records).set_index(df.index)
                df[cols] = post[cols]

        return abc

    def prepare_dataset(self, setting: Setting) -> OnePairDataset:
        pair = (setting.src_lang, setting.tgt_lang)
        if pair not in self._pair2dfs:
            raise RuntimeError(f'Pair {pair} not added.')
        # Get alphabets first.

        def check_abc(lang: Lang):
            if lang not in self._lang2abc:
                raise RuntimeError(f'Alphabet for {lang} has not been prepared.')

        check_abc(setting.src_lang)
        check_abc(setting.tgt_lang)

        # Get relevant data frames.
        src_df, tgt_df = self._pair2dfs[pair]
        src_df = setting.split.select(src_df)
        tgt_df = setting.split.select(tgt_df)
        pair_df = pd.merge(src_df, tgt_df,
                           left_index=True, right_index=True,
                           suffixes=('_src', '_tgt'))
        if setting.keep_ratio is not None:
            logging.imp(f'keep_ratio is {setting.keep_ratio}.')
            num = int(len(pair_df) * setting.keep_ratio)
            pair_df = pair_df.sample(num, random_state=g.random_seed)
        vc = pair_df.index.value_counts()
        vc.name = 'num_variants'
        pair_df = pd.merge(pair_df, vc, left_index=True, right_index=True)
        pair_df['sample_weight'] = 1 / vc

        logging.info(f'Total number of cognate pairs for {pair} for {setting.split}: {len(pair_df)}.')

        return OnePairDataset.from_dataframe(setting.src_lang, setting.tgt_lang, pair_df)

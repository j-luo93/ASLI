import pickle
import unicodedata
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from dev_misc import NDA

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from ipapy.ipastring import IPAString
from lingpy.sequence.sound_classes import ipa2tokens
from networkx.algorithms.components import connected_components
from networkx.algorithms.shortest_paths import shortest_path
from tqdm import tqdm

from dev_misc.utils import ErrorRecord, recorded_try
from pypheature.nphthong import InvalidNphthong, Nphthong
from pypheature.process import (FeatureProcessor, InvalidBaseSegment,
                                NoMappingFound, NonUniqueMapping, Segment)
from pypheature.segment import ExclusivityFailure, InvalidSegment, Segment

PDF = pd.DataFrame


def run_section(before_msg: str, after_msg: str, **kwargs):

    def decorator(func):

        cached_func = st.cache(hash_funcs={PDF: id, Segment: id, Nphthong: id, FeatureProcessor: id}, **kwargs)(func)

        @wraps(cached_func)
        def wrapped(*args, **kwargs):
            status_text = st.subheader(before_msg)
            ret = cached_func(*args, **kwargs)
            status_text.subheader(before_msg + '\t' + after_msg)
            return ret

        return wrapped

    return decorator


@run_section('Loading data...', 'Loading done.')
def load_data(path: str) -> PDF:
    return pd.read_csv(path, sep='\t', error_bad_lines=False)


def run_with_argument(name: str, *, parser: Optional[ArgumentParser] = None, default: Optional[Any] = None, msg: Optional[str] = None):
    if st._is_running_with_streamlit:
        argument = st.text_input(f'{name}:', default, help=msg)
        return argument
    else:
        assert parser is not None, 'Must pass parser for script mode.'
        parser.add_argument(f'--{name}', default=default, help=msg)
        args = parser.parse_known_args()[0]
        return getattr(args, name)


class I2tException(Exception):
    """Raise this when you have any `i2t` issue."""


def i2t(ipa: str) -> List[str]:
    """ipa2token call. Raises error if return is empty."""
    ret = ipa2tokens(ipa, merge_vowels=True, merge_geminates=False)
    if not ret:
        raise I2tException
    return ret


def recorded_assign(df: PDF, new_name: str, old_name: str, func, error_cls=AssertionError) -> Tuple[PDF, PDF]:
    errors = list()
    df = df.assign(**{new_name: recorded_try(df, old_name, func, error_cls, errors=errors)})
    error_df = df.iloc[[error.idx for error in errors]]
    return df, error_df


def standardize(ph: str) -> str:
    return unicodedata.normalize('NFD', ph)


def fv_func(seg):
    if isinstance(seg, Segment):
        return (tuple([(k, str(v.value)) for k, v in sorted(asdict(seg).items()) if k not in ['ipa', 'diacritics', 'base']]), )

    ret = tuple()
    for v in seg.vowels:
        ret += fv_func(v)
    return ret

# # Compute distance matrix.


@lru_cache(maxsize=None)
def get_sub_cost(seg1: Segment, seg2: Segment, quiet: bool = True) -> float:
    cost = 0.0

    def helper(name, weight):
        v1 = getattr(seg1, name).value
        v2 = getattr(seg2, name).value
        ret = (v1 is not v2) * weight
        if not quiet:
            print(f'{name} cost', ret, 'from', v1, 'and', v2)
        return ret

    # syllabic, consonantal, approximant, sonorant for sonority hierarchy.
    for name in ['syllabic', 'consonantal', 'approximant', 'sonorant']:
        cost += helper(name, 1.0 / 4)
    # continuant and delayed_release for different obstruents, but delayed_release is optional.
    if seg1.is_obstruent() and seg2.is_obstruent():
        cost += helper('continuant', 0.5 / 2)
        cost += helper('delayed_release', 0.5 / 2)
    # trill and tap
    if seg1.is_liquid() and seg2.is_liquid():
        cost += helper('trill', 0.5 / 2)
        cost += helper('tap', 0.5 / 2)
    # dorsal features
    cost += helper('dorsal', 1.0 / 3)
    if seg1.is_dorsal() and seg2.is_dorsal():
        for name in ['high', 'low', 'front', 'back']:
            cost += helper(name, 0.5 / 4)
    # coronal features
    cost += helper('coronal', 1.0 / 3)
    if seg1.is_coronal() and seg2.is_coronal():
        for name in ['anterior', 'distributed', 'strident', 'lateral']:
            cost += helper(name, 0.5 / 4)
    # labial features.
    cost += helper('labial', 1.0 / 3)
    if seg1.is_labial() and seg2.is_labial():
        for name in ['labiodental', 'round']:
            cost += helper(name, 0.5 / 2)

    # laryngeal features
    for name in ['voice', 'spread_glottis', 'constricted_glottis']:
        cost += helper(name, 0.25)
    # some minor features.
    for name in ['tense', 'round', 'long', 'nasal', 'overlong']:
        cost += helper(name, 0.25)

    return cost


def iter_seg(seg):
    if isinstance(seg, Segment):
        yield seg
    else:
        yield from seg.vowels


def edit_dist(seg1, seg2, ins_cost=1):

    def get_len(seg):
        return 1 if isinstance(seg, Segment) else len(seg)

    l1 = get_len(seg1)
    l2 = get_len(seg2)
    subcost = np.zeros([l1, l2], dtype='float32')
    for i, s1 in enumerate(iter_seg(seg1)):
        for j, s2 in enumerate(iter_seg(seg2)):
            subcost[i, j] = get_sub_cost(s1, s2)

    l = min(l1, l2)
    dist = np.full([l1 + 1, l2 + 1, l + 1], 10000, dtype='float32')
    for i in range(l1 + 1):
        dist[i, 0, 0] = i * ins_cost
    for j in range(l2 + 1):
        dist[0, j, 0] = j * ins_cost

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            for k in range(1, l + 1):
                dist[i, j, k] = min(dist[i - 1, j - 1, k - 1] + subcost[i - 1, j - 1],
                                    min(dist[i - 1, j, k], dist[i, j - 1, k]) + ins_cost)
    return dist[l1, l2, l]


@run_section('Adding phones specific to our dataset...', 'Phones added.')
def add_phones(df: PDF, added_phones: List[str]) -> PDF:
    df = df['rawIPA'].str.split().explode().reset_index().rename(columns={'index': 'raw_word_id'})
    df = df.dropna(subset=['rawIPA'])

    for ph in added_phones:
        # ipa2tokens cannot properly deal with some overlong vowels.
        df = df.append({
            'raw_word_id': len(df),
            'rawIPA': (f'{ph} ' * 100).strip()},
            ignore_index=True)
    return df


@run_section('Removing stress from transcriptions...', 'Removal done.')
def destress(df: PDF) -> PDF:
    return df.assign(rawIPA=df['rawIPA'].apply(lambda ipa: ipa.replace('ˌ', '').replace('ˈ', '')))


@run_section('Tokenizing transcriptions...', 'Tokenization done.')
def tokenize(df: PDF) -> Tuple[PDF, PDF, str]:
    old_name = 'rawIPA'
    df, error_df = recorded_assign(df, 'raw_toks', old_name, i2t, I2tException)
    df = df.dropna(subset=['raw_toks'])
    return df, error_df, old_name


@run_section('Standardizing phones...', 'Standardization done.')
def standardize_phones(df: PDF) -> PDF:
    raw_ph_df = PDF(set(df.explode('raw_toks')['raw_toks']), columns=['raw_ph'])
    return raw_ph_df.assign(std_ph=raw_ph_df['raw_ph'].apply(standardize))


@run_section('Obtain feature vectors using `pypheature`...', 'Feature vectors obtained.')
def get_feature_vectors(df: PDF, processor: FeatureProcessor) -> Tuple[PDF, PDF, str]:
    std_ph_df = PDF(set(df['std_ph']), columns=['std_ph'])
    assert len(set(std_ph_df['std_ph'])) == len(std_ph_df)

    old_name = 'std_ph'
    std_ph_df, error_df = recorded_assign(std_ph_df, 'segment', old_name, processor.process,
                                          (InvalidBaseSegment, InvalidNphthong, InvalidSegment, ExclusivityFailure))
    std_ph_df = std_ph_df.dropna().reset_index(drop=True)
    std_ph_df = std_ph_df.assign(fv=std_ph_df['segment'].apply(fv_func))
    return std_ph_df, error_df, old_name


@run_section('Getting the prototypical sounds -- merging sounds into one if they are identical based on the feature vectors...',
             'Merging done.')
def get_proto_phones(std_ph_df: PDF, raw_ph_df: PDF, words_df: PDF) -> PDF:
    std_ph_lst = std_ph_df.pivot_table(index='fv', values='std_ph', aggfunc=list)
    raw2std = raw_ph_df[['raw_ph', 'std_ph']].set_index('raw_ph', verify_integrity=True).to_dict()['std_ph']
    std2cnt = words_df['raw_toks'].explode().apply(raw2std.get).value_counts().to_dict()

    def get_proto_ph(lst: List[str]) -> str:
        # You want the segment with the highest count, and then shortest length.
        stats = [(std2cnt[seg], -len(standardize(seg))) for seg in lst]
        max_stat = max(stats)
        return lst[stats.index(max_stat)]

    std_ph_lst = std_ph_lst.assign(proto_ph=std_ph_lst['std_ph'].apply(get_proto_ph))
    std2proto = std_ph_lst.explode('std_ph').set_index('std_ph').to_dict()['proto_ph']
    merged_cnt = pd.merge(PDF(std2cnt.items(), columns=['std_ph', 'cnt']),
                          PDF(std2proto.items(), columns=['std_ph', 'proto_ph']),
                          left_on='std_ph', right_on='std_ph', how='inner')
    return merged_cnt


def show_errors(error_df: PDF, old_name: str):
    st.write(f'{len(error_df)} errors in total, results computed from `{old_name}` column:')
    st.write(error_df)


@run_section('Getting phones to keep (frequency >= 50)...', 'Done.')
def get_kept_phones(df: PDF, processor: FeatureProcessor) -> Tuple[PDF, List[int], Dict[str, int], List[int], List[Union[Segment, Nphthong]]]:
    proto2cnt = df.pivot_table(index='proto_ph', values='cnt',
                               aggfunc='sum').sort_values('cnt', ascending=False)

    i2pp = list(proto2cnt.index)
    pp2i = {pp: i for i, pp in enumerate(i2pp)}
    kept_ids = [i for pp, i in pp2i.items() if proto2cnt.loc[pp]['cnt'] >= 50]
    segments = [processor.process(pp) for pp in i2pp]
    return proto2cnt, i2pp, pp2i, kept_ids, segments


@run_section('Loading feature processor...', 'Loading done.')
def load_processor() -> FeatureProcessor:
    return FeatureProcessor()


@run_section('Getting edit dist between prototypes...', 'Computation done.', suppress_st_warning=True)
def get_edit_dist(i2pp: List[str], segments: List[Union[Segment, Nphthong]], insert_cost: float) -> NDA:
    dist_mat = np.zeros([len(i2pp), len(i2pp)], dtype='float32')
    pbar = st.progress(0.0)
    pbar_status = st.empty()
    for i, seg1 in tqdm(enumerate(segments)):
        for j, seg2 in enumerate(segments):
            dist_mat[i, j] = edit_dist(seg1, seg2, ins_cost=insert_cost)
            pbar.progress(i / len(segments))
        pbar_status.text(f'{i + 1} / {len(segments)} = {((i + 1) / len(segments) * 100.0):.1f}% done.')
    return dist_mat


def should_proceed(key: str) -> bool:
    if st._is_running_with_streamlit:
        return st.radio('Proceed?', ['Yes', 'No'], index=1, key=key) == 'Yes'
    return True


def get_connected_sounds(ph, g, kept_dist_mat, kept_i2pp, kept_pp2i) -> PDF:
    i = kept_pp2i[ph]
    ret = list()
    for u, v in g.edges(i):
        sound = kept_i2pp[v]
        ret.append((sound, float(kept_dist_mat[i, v])))
    return PDF(ret, columns=['IPA', 'distance'])


if __name__ == "__main__":
    parser = ArgumentParser()
    st.title('Prepare alphabet.')
    st.header('Specify the arguments first:')
    data_path = run_with_argument('data_path', parser=parser,
                                  default='data/northeuralex-0.9-forms.tsv', msg="Path to the NorthEuraLex dataset.")
    raw_words_df = load_data(data_path)

    # Add some phones to the dataset -- they might not be present in the original data.
    added_phones = ['oːː', 'eːː', 'õː', 'ĩː', 'xʷ', 'gʷ', 'hʷ', 'ay', 'iuː', 'ioː',
                    'io', 'eːo', 'æa', 'æːa', 'eo', 'iːu', 'iu', 'ɣː', 'ðː', 'wː', 'θː', 'βː', 'øy']
    words_df = add_phones(raw_words_df, added_phones)
    st.write(f'{", ".join(added_phones)}')

    words_df = destress(words_df)

    words_df, error_df, old_name = tokenize(words_df)
    show_errors(error_df, old_name)

    if should_proceed('tokenized'):
        raw_ph_df = standardize_phones(words_df)
        processor = load_processor()
        std_ph_df, error_df, old_name = get_feature_vectors(raw_ph_df, processor)
        show_errors(error_df, old_name)

        if should_proceed('standardized'):
            merged_cnt = get_proto_phones(std_ph_df, raw_ph_df, words_df)
            proto2cnt, i2pp, pp2i, kept_ids, segments = get_kept_phones(merged_cnt, processor)
            st.write(f'{len(kept_ids)} sounds are kept.')
            insert_cost = 0.5
            dist_mat = get_edit_dist(i2pp, segments, insert_cost)

            kept_dist_mat = dist_mat[np.asarray(kept_ids).reshape(-1, 1), kept_ids]
            kept_i2pp = [i2pp[i] for i in kept_ids]
            kept_pp2i = {pp: i for i, pp in enumerate(kept_i2pp)}

            # Building graphs of connection.
            top_k = 10
            g = nx.Graph()
            for i, pp in enumerate(kept_i2pp):
                g.add_node(i)
                # There are two ways of adding an edge. First, the distance is <= insert_cost and it falls within the top k neighbors.
                sort_i = kept_dist_mat[i, :].copy().argsort()
                dists = kept_dist_mat[i, sort_i]
                # Since the closest sound is always itself, we need to use index top_k, instead of top_k - 1.
                max_dist = max(dists[top_k], insert_cost)
                for j, d in zip(sort_i, dists):
                    if d > max_dist:
                        break
                    if i != j:
                        g.add_edge(i, j)
                # Second, you might add a new vowel for nphthongs.
                seg = segments[pp2i[pp]]
                if isinstance(seg, Segment) and seg.is_vowel():
                    for j, pp in enumerate(kept_i2pp):
                        seg_j = segments[pp2i[pp]]
                        if isinstance(seg_j, Nphthong) and len(seg_j) == 2 and kept_dist_mat[i, j] == insert_cost:
                            g.add_edge(i, j)
                elif isinstance(seg, Nphthong):
                    for j, pp in enumerate(kept_i2pp):
                        seg_j = segments[pp2i[pp]]
                        if isinstance(seg_j, Segment) and seg_j.is_vowel() and kept_dist_mat[i, j] == insert_cost:
                            g.add_edge(i, j)
                        elif isinstance(seg_j, Nphthong) and abs(len(seg) - len(seg_j)) == 1 and kept_dist_mat[i, j] == insert_cost:
                            g.add_edge(i, j)

            query_sound = st.selectbox('Query sound', sorted(kept_i2pp))
            st.write(get_connected_sounds(query_sound, g, kept_dist_mat, kept_i2pp, kept_pp2i))

            cc = list(connected_components(g))
            assert len(cc) == 1

            # Compute average number of connected sounds.
            cnt = dict()
            for i in kept_ids:
                cnt[i2pp[i]] = len(g.edges(i))
            st.write(f'Average number of connected sounds: {(sum(cnt.values()) / len(kept_ids)):.3f}')

            if should_proceed('about_to_save'):
                proto_ph_map = dict()
                for i in kept_ids:
                    ph = i2pp[i]
                    proto_ph_map[ph] = ph

                lengths = [len(pp) for pp in kept_i2pp]
                neg_counts = [-proto2cnt.loc[pp]['cnt'] for pp in kept_i2pp]
                for i, pp in tqdm(enumerate(i2pp)):
                    if i not in kept_ids:
                        dists = dist_mat[i, kept_ids].copy()
                        stats = list(zip(dists, neg_counts, lengths))
                        j = stats.index(min(stats))
                        proto_ph_map[pp] = kept_i2pp[j]

                g_edges = set(g.edges)
                edges = set(g_edges)
                for i, j in g_edges:
                    edges.add((j, i))

                for i in range(len(i2pp)):
                    assert (i, i) not in edges

                processor.load_repository(kept_i2pp)

                cl_map = dict()
                gb_map = dict()
                for ph in kept_i2pp:
                    segment = processor.process(ph)
                    if isinstance(segment, Segment) and segment.is_short():
                        try:
                            after = processor.change_features(segment, ['+long'])
                        except NonUniqueMapping:
                            print(f"non-unique mapping for {ph}")
                        except NoMappingFound:
                            print(f"no mapping for {ph}")

                        after_id = kept_pp2i[str(after)]
                        before_id = kept_pp2i[ph]
                        if (before_id, after_id) in edges:
                            print(ph, after)
                            cl_map[ph] = str(after)
                    if isinstance(segment, Nphthong) and len(segment.vowels) == 2:
                        first = str(segment.vowels[0])
                        second = str(segment.vowels[1])
                        if first in ['i', 'u'] and second in kept_pp2i:
                            gb_map[ph] = second

                out_path = 'data/nel_segs.pkl'
                with open(out_path, 'wb') as fout:
                    pickle.dump({
                        'proto_ph_map': proto_ph_map,
                        'proto_ph_lst': kept_i2pp,
                        'dist_mat': kept_dist_mat,
                        'edges': [(kept_i2pp[i], kept_i2pp[j]) for i, j in edges],
                        'cl_map': cl_map,
                        'gb_map': gb_map},
                        fout)
                st.write(f'Saved to {out_path}.')

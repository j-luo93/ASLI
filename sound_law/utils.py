from argparse import ArgumentParser
from functools import wraps
from typing import Any, Optional

import pandas as pd
import streamlit as st

from pypheature.nphthong import Nphthong
from pypheature.process import FeatureProcessor
from pypheature.segment import Segment

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


def run_with_argument(name: str, *, parser: Optional[ArgumentParser] = None, default: Optional[Any] = None, msg: Optional[str] = None):
    if st._is_running_with_streamlit:
        argument = st.text_input(f'{name}:', default, help=msg)
        return argument
    else:
        assert parser is not None, 'Must pass parser for script mode.'
        parser.add_argument(f'--{name}', default=default, help=msg)
        args = parser.parse_known_args()[0]
        return getattr(args, name)

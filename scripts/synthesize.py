from argparse import ArgumentParser
from typing import Callable, Tuple

import pandas as pd

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('src_path', type=str, help='Path to the src tsv file.')
    parser.add_argument('out_path', type=str, help='Output path.')
    parser.add_argument('mode', type=str, help='Configuration name.')
    args = parser.parse_args()

    converter = dispatch[args.mode]

    df = pd.read_csv(args.src_path, sep='\t', keep_default_na=True, error_bad_lines=False)
    out = df.copy()
    with open(args.out_path, 'w') as fout:
        out['transcription'] = out['transcription'].apply(converter.form_change)
        out['ipa'] = out['ipa'].apply(converter.ipa_change)
        out['tokens'] = out['tokens'].apply(converter.ipa_change)
    out.to_csv(args.out_path, sep='\t', index=None)

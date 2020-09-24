import sys
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation
from functools import lru_cache
from typing import List, Optional

import altair as alt
# import gradio as gr
import numpy as np
import pandas as pd
import torch
# from gradio.inputs import Dropdown, Textbox
from loguru import logger
from prompt_toolkit import PromptSession

from dev_misc.arglib import set_argument
from dev_misc.trainlib import has_gpus
from sound_law.config import all_germanic_langs, all_italic_langs
from sound_law.data.alphabet import EOT
from sound_law.data.data_loader import SourceOnlyBatch
from sound_law.main import main, setup
from sound_law.train.manager import OneToManyManager

manager = None


@lru_cache(maxsize=None)
def get_manager(saved_g_path: str, saved_model_path: str) -> OneToManyManager:
    # HACK(j_luo)
    global manager
    # Get the manager first.
    initiator = setup()
    initiator.run(saved_g_path=saved_g_path)
    logger.info('Initiator ran.')
    manager = OneToManyManager()
    logger.info('Initialized the manager.')

    # The value of `saved_model_path` in the saved `g` is not the model we want to load. Use the passed value instead.
    state_dict = torch.load(saved_model_path)
    manager.model.load_state_dict(state_dict)
    logger.info('Model loaded.')
    set_argument('comp_mode', 'units', _force=True)
    return manager


def translate_kernel(ipa_tokens: str, tgt_lang: str):
    global manager
    tgt_lang_id = manager.lang2id[tgt_lang]
    batch = SourceOnlyBatch.from_ipa_tokens(ipa_tokens, manager.src_abc, tgt_lang_id)
    if has_gpus():
        batch.cuda()

    manager.model.eval()
    with torch.no_grad():
        hyps = manager.model.predict(batch)
    preds, pred_lengths = hyps.translate(manager.tgt_abc)
    preds = preds[0]
    pred_lengths = pred_lengths[0]

    def get_pred_unit(pred: List[str], ind: int) -> str:
        if ind < len(pred):
            return pred[ind]
        if ind != len(pred):
            raise RuntimeError('Miscalculation of prediction length?')
        return EOT

    beam_id = 0
    pred = preds[beam_id]
    src_units = batch.src_seqs.units[0]
    src_l = len(src_units)
    tgt_l = pred_lengths[beam_id]
    almt = hyps.almt[0, beam_id].t().detach().cpu().numpy()[:src_l, :tgt_l]

    x, y = np.meshgrid(range(src_l), range(tgt_l), indexing='ij')
    df = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'almt': almt.ravel()})
    df['x_text'] = df['x'].apply(lambda i: src_units[i])
    df['y_text'] = df['y'].apply(lambda i: get_pred_unit(pred, i))
    df['x_ou'] = df[['x', 'x_text']].apply(lambda item: f'{item[0]}-{item[1]}', axis=1)
    df['y_ou'] = df[['y', 'y_text']].apply(lambda item: f'{item[0]}-{item[1]}', axis=1)
    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X('x_ou', type='nominal', sort=sorted(df['x_ou'].unique(), key=lambda s: int(s.split('-')[0]))),
        y=alt.Y('y_ou:O', sort=alt.EncodingSortField('y', order='descending')),
        color='almt:Q',
        tooltip=['x_text', 'y_text', 'almt']
    ).configure_view(step=100).configure_axis(labelFontSize=20, titleFontSize=20).configure_axisBottom(labelAngle=0)
    heatmap.save('chart.html')

    return heatmap


name2func = dict()


def command(cmd_name: str, short_name: Optional[str] = None):
    def check_name(name: str):
        if name in name2func:
            raise NameError(f'Duplicate name {name}.')
    check_name(cmd_name)
    check_name(short_name)

    def decorator(func):
        name2func[cmd_name] = func
        if short_name:
            name2func[short_name] = func
        return func

    return decorator


tgt_lang = None


@command('set-lang', 'set')
def set_lang(lang: str):
    global tgt_lang
    tgt_lang = lang


@command('translate', 'tr')
def translate(ipa_tokens: str):
    global tgt_lang
    ipa_tokens = ipa_tokens.replace(',', ' ')
    translate_kernel(ipa_tokens, tgt_lang)


class CommandNameNotFound(Exception):
    """Raise this error if command name is not found."""


def eval_prompt(prompt: str):
    cmd_name, *args = prompt.split()
    try:
        cmd = name2func[cmd_name]
    except KeyError:
        raise CommandNameNotFound()
    cmd(*args)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('ip', type=str, help='IP for serving gradio.')
    parser.add_argument('config_file', type=str, help='Path to the config file.')
    args = parser.parse_args()

    cfg_parser = ConfigParser(allow_no_value=True,
                              interpolation=ExtendedInterpolation())
    cfg_parser.read(args.config_file)

    if 'model' not in cfg_parser:
        raise ValueError('The configuration file must have a model section.')

    saved_g_path = cfg_parser['model']['saved_g_path']
    saved_model_path = cfg_parser['model']['saved_model_path']

    # ipa_seq_box = Textbox(placeholder="Input your tokenized IPA sequence here.",
    #                       default='a b a')
    # lang_box = Textbox(placeholder='target language')

    # io = gr.Interface(fn=translate,
    #                   inputs=[ipa_seq_box, lang_box],
    #                   outputs="html",
    #                   server_name=args.ip)

    # HACK(j_luo) I'm using the main thread for loading the manager. For some reason, using a child thread is extremely slow.
    get_manager(saved_g_path, saved_model_path)

    session = PromptSession()
    while True:
        try:
            text = session.prompt('> ')
            eval_prompt(text)
            print('Evaluation finished.')
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except:
            print('Failed.')

    # io.launch()
    # get_manager(saved_g_path, saved_model_path)

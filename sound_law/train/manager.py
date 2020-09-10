"""
A manager class takes care of managing the data loader, the model, and the trainer.
"""
import logging
from typing import Optional

import torch
from torch.optim import Adam

from dev_misc import add_argument, add_condition, g, get_tensor
from dev_misc.devlib.helper import has_gpus
from dev_misc.trainlib.tb_writer import MetricWriter
from sound_law.data.alphabet import Alphabet
from sound_law.data.cognate import CognateRegistry, get_paths
from sound_law.data.data_loader import DataLoaderRegistry
from sound_law.data.dataset import Split
from sound_law.data.setting import Setting
from sound_law.evaluate.evaluator import Evaluator
from sound_law.model.one_pair import OnePairModel
from sound_law.model.one_to_many import OneToManyModel

from .trainer import Trainer

add_argument('check_interval', default=10, dtype=int, msg='Frequency to check the training progress.')
add_argument('eval_interval', default=100, dtype=int, msg='Frequency to call the evaluator.')
add_argument('save_interval', dtype=int, msg='Frequency to save the progress and the model.')
add_argument('keep_ratio', dtype=float, msg='Ratio of cognate pairs to keep.')
add_argument('test_keep_ratio', dtype=float, msg='Ratio of cognate pairs to keep for the test target language.')
add_argument('train_e_keep_ratio', dtype=float,
             msg='Ratio of cognate pairs to keep for the training set during evaluation.')
add_argument('saved_model_path', dtype='path', msg='Path to the saved model.')
add_argument('evaluate_only', dtype=bool, default=False, msg='Flag to toggle evaluate-only mode.')
add_argument('share_src_tgt_abc', dtype=bool, default=False, msg='Flag to share the alphabets for source and target.')
add_argument('use_phono_features', dtype=bool, default=False, msg='Flag to use phonological features.')
add_condition('use_phono_features', True, 'share_src_tgt_abc', True)


class OnePairManager:
    """A manager for sound law induction on one src-tgt pair."""

    def __init__(self):
        # Prepare cognate registry.
        cr = CognateRegistry()
        cr.add_pair(g.data_path, g.src_lang, g.tgt_lang)

        # Prepare alphabets now.
        if g.share_src_tgt_abc:
            self.src_abc = cr.prepare_alphabet(g.src_lang, g.tgt_lang)
            self.tgt_abc = self.src_abc
        else:
            self.src_abc = cr.prepare_alphabet(g.src_lang)
            self.tgt_abc = cr.prepare_alphabet(g.tgt_lang)

        # Prepare data loaders with different splits.
        self.dl_reg = DataLoaderRegistry()

        def create_setting(name: str, split: Split, for_training: bool, keep_ratio: Optional[float] = None) -> Setting:
            return Setting(name, 'one_pair', split, g.src_lang, g.tgt_lang, for_training, keep_ratio=keep_ratio)

        def register_dl(setting: Setting):
            self.dl_reg.register_data_loader(setting, cr)

        if g.input_format == 'wikt':
            train_setting = create_setting('train', Split('train'), True,
                                           keep_ratio=g.keep_ratio)
            train_e_setting = create_setting('train_e', Split('train'), False,
                                             keep_ratio=g.train_e_keep_ratio)  # For evaluation.
            dev_setting = create_setting('dev', Split('dev'), False)
            register_dl(train_setting)
            register_dl(train_e_setting)
            register_dl(dev_setting)
        else:
            for fold in range(5):
                dev_fold = fold + 1
                train_folds = list(range(1, 6))
                train_folds.remove(dev_fold)

                train_setting = create_setting(
                    f'train@{fold}', Split('train', train_folds), True,
                    keep_ratio=g.keep_ratio)
                train_e_setting = create_setting(
                    f'train@{fold}_e', Split('train', train_folds), False,
                    keep_ratio=g.train_e_keep_ratio)
                dev_setting = create_setting(f'dev@{fold}', Split('dev', [dev_fold]), False)
                register_dl(train_setting)
                register_dl(train_e_setting)
                register_dl(dev_setting)

        test_setting = create_setting('test', Split('test'), False)
        register_dl(test_setting)

    def run(self):
        phono_feat_mat = special_ids = None
        if g.use_phono_features:
            phono_feat_mat = get_tensor(self.src_abc.pfm)
            special_ids = get_tensor(self.src_abc.special_ids)

        metric_writer = MetricWriter(g.log_dir, flush_secs=5)

        def run_once(train_name, dev_name, test_name):
            train_dl = self.dl_reg[train_name]
            train_e_dl = self.dl_reg[f'{train_name}_e']
            dev_dl = self.dl_reg[dev_name]
            test_dl = self.dl_reg[test_name]

            model = OnePairModel(len(self.src_abc), len(self.tgt_abc),
                                 phono_feat_mat=phono_feat_mat,
                                 special_ids=special_ids)

            if g.saved_model_path is not None:
                model.load_state_dict(torch.load(g.saved_model_path, map_location=torch.device('cpu')))
                logging.imp(f'Loaded from {g.saved_model_path}.')
            if has_gpus():
                model.cuda()
            logging.info(model)

            evaluator = Evaluator(model, {train_name: train_e_dl, dev_name: dev_dl, test_name: test_dl},
                                  self.tgt_abc,
                                  metric_writer=metric_writer)

            if g.evaluate_only:
                # FIXME(j_luo) load global_step from saved model.
                evaluator.evaluate('evaluate_only', 0)
            else:
                trainer = Trainer(model, [self.dl_reg.get_setting_by_name(train_name)],
                                  [1.0], 'step',
                                  stage_tnames=['step'],
                                  check_interval=g.check_interval,
                                  evaluator=evaluator,
                                  eval_interval=g.eval_interval,
                                  save_interval=g.save_interval,
                                  metric_writer=metric_writer)
                if g.saved_model_path is None:
                    trainer.init_params('uniform', -0.1, 0.1)
                trainer.set_optimizer(Adam, lr=0.002)
                trainer.train(self.dl_reg)

        if g.input_format == 'wikt':
            run_once('train', 'dev', 'test')
        else:
            for fold in range(5):
                logging.imp(f'Cross-validation, fold number {fold}')
                run_once(f'train@{fold}', f'dev@{fold}', 'test')


class OneToManyManager:
    """The manager class for single-source-multiple-target scenarios."""

    add_argument('train_tgt_langs', dtype=str, nargs='+', msg='Target languages used for training.')

    def __init__(self):
        # Prepare cognate registry first.
        cr = CognateRegistry()
        all_tgt = sorted([g.tgt_lang] + list(g.train_tgt_langs))
        for tgt in all_tgt:
            cr.add_pair(g.data_path, g.src_lang, tgt)

        # Get alphabets. Note that the target alphabet is based on the union of all target languages, i.e., a shared alphabet for all.
        if g.share_src_tgt_abc:
            self.src_abc = cr.prepare_alphabet(*(all_tgt + [g.src_lang]))
            self.tgt_abc = self.src_abc
        else:
            self.src_abc = cr.prepare_alphabet(g.src_lang)
            self.tgt_abc = cr.prepare_alphabet(*all_tgt)

        # Get stats for unseen units.
        stats = self.tgt_abc.stats
        _, test_tgt_path = get_paths(g.data_path, g.src_lang, g.tgt_lang)
        mask = (stats.sum() == stats.loc[test_tgt_path])
        unseen = mask[mask].index.tolist()
        total = len(stats.loc[test_tgt_path].dropna())
        logging.info(f'Unseen units ({len(unseen)}/{total}) for {g.tgt_lang} are: {unseen}.')

        # Get language-to-id mappings. Used only for the targets (i.e., decoder side).
        lang2id = {tgt: i for i, tgt in enumerate(all_tgt)}

        # Get all data loaders.
        self.dl_reg = DataLoaderRegistry()

        def create_setting(name: str, tgt_lang: str, split: Split, for_training: bool, keep_ratio: Optional[float] = None) -> Setting:
            return Setting(name, 'one_pair', split, g.src_lang, tgt_lang, for_training, keep_ratio=keep_ratio)

        def register_dl(setting: Setting):
            self.dl_reg.register_data_loader(setting, cr, lang2id=lang2id)

        test_setting = create_setting(f'test@{g.tgt_lang}', g.tgt_lang, Split('all'), False,
                                      keep_ratio=g.test_keep_ratio)
        register_dl(test_setting)

        # Get the training languages.
        for train_tgt_lang in g.train_tgt_langs:
            if g.input_format == 'ielex':
                train_split = Split('train', [1, 2, 3, 4])  # Use the first four folds for training.
                dev_split = Split('dev', [5])  # Use the last fold for dev.
            else:
                train_split = Split('train')
                dev_split = Split('dev')
            train_setting = create_setting(f'train@{train_tgt_lang}', train_tgt_lang,
                                           train_split, True, keep_ratio=g.keep_ratio)
            train_e_setting = create_setting(
                f'train@{train_tgt_lang}_e', train_tgt_lang, train_split, False, keep_ratio=g.train_e_keep_ratio)
            dev_setting = create_setting(f'dev@{train_tgt_lang}', train_tgt_lang, dev_split, False)
            test_setting = create_setting(f'test@{train_tgt_lang}', train_tgt_lang, Split('test'), False)

            register_dl(train_setting)
            register_dl(train_e_setting)
            register_dl(dev_setting)
            register_dl(test_setting)

        phono_feat_mat = special_ids = None
        if g.use_phono_features:
            phono_feat_mat = get_tensor(self.src_abc.pfm)
            special_ids = get_tensor(self.src_abc.special_ids)

        self.model = OneToManyModel(len(self.src_abc), len(self.tgt_abc),
                                    len(g.train_tgt_langs) + 1, lang2id[g.tgt_lang],
                                    lang2id=lang2id,
                                    phono_feat_mat=phono_feat_mat,
                                    special_ids=special_ids)

        if g.saved_model_path is not None:
            self.model.load_state_dict(torch.load(g.saved_model_path, map_location=torch.device('cpu')))
            logging.imp(f'Loaded from {g.saved_model_path}.')
        if has_gpus():
            self.model.cuda()
        logging.info(self.model)

        metric_writer = MetricWriter(g.log_dir, flush_secs=5)

        # NOTE(j_luo) Evaluate on every loader that is not for training.
        eval_dls = self.dl_reg.get_loaders_by_name(lambda name: 'train' not in name or '_e' in name)
        self.evaluator = Evaluator(self.model, eval_dls, self.tgt_abc,
                                   metric_writer=metric_writer)

        if not g.evaluate_only:
            train_names = [f'train@{train_tgt_lang}' for train_tgt_lang in g.train_tgt_langs]
            train_settings = [self.dl_reg.get_setting_by_name(name) for name in train_names]
            self.trainer = Trainer(self.model, train_settings,
                                   [1.0] * len(train_settings), 'step',
                                   stage_tnames=['step'],
                                   evaluator=self.evaluator,
                                   check_interval=g.check_interval,
                                   eval_interval=g.eval_interval,
                                   save_interval=g.save_interval,
                                   metric_writer=metric_writer)
            if g.saved_model_path is None:
                self.trainer.init_params('uniform', -0.1, 0.1)
            self.trainer.set_optimizer(Adam, lr=0.002)

    def run(self):
        if g.evaluate_only:
            # FIXME(j_luo) load global_step from saved model.
            self.evaluator.evaluate('evaluate_only', 0)
        else:
            self.trainer.train(self.dl_reg)

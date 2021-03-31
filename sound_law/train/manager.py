"""
A manager class takes care of managing the data loader, the model, and the trainer.
"""
import logging
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.optim import SGD, Adam

from dev_misc import add_argument, add_condition, g, get_tensor
from dev_misc.devlib.helper import has_gpus
from dev_misc.trainlib.tb_writer import MetricWriter
from sound_law.data.alphabet import (ANY_ID, ANY_S_ID, ANY_UNS_ID, EMP_ID,
                                     EOT_ID, NULL_ID, SOT_ID, SYL_EOT_ID,
                                     Alphabet)
from sound_law.data.cognate import CognateRegistry, get_paths
from sound_law.data.data_loader import DataLoaderRegistry
from sound_law.data.setting import Setting, Split
from sound_law.evaluate.evaluator import Evaluator
from sound_law.rl.action import SoundChangeAction
from sound_law.rl.agent import A2C, VanillaPolicyGradient
from sound_law.rl.env import SoundChangeEnv  # , TrajectoryCollector
from sound_law.rl.mcts import Mcts
# pylint: disable=no-name-in-module
from sound_law.rl.mcts_cpp import (  # pylint: disable=no-name-in-module
    PyActionSpaceOpt, PyEnv, PyEnvOpt, PyWordSpaceOpt, PyMctsOpt)
# pylint: enable=no-name-in-module
from sound_law.rl.trajectory import VocabState
from sound_law.s2s.module import CharEmbedding, EmbParams, PhonoEmbedding
from sound_law.s2s.one_pair import OnePairModel
from sound_law.s2s.one_to_many import OneToManyModel

from .trainer import MctsTrainer, Trainer
# from .trainer import MctsTrainer, PolicyGradientTrainer, Trainer

add_argument('batch_size', default=32, dtype=int, msg='Batch size.')
add_argument('check_interval', default=10, dtype=int, msg='Frequency to check the training progress.')
add_argument('eval_interval', default=100, dtype=int, msg='Frequency to call the evaluator.')
add_argument('save_interval', dtype=int, msg='Frequency to save the progress and the model.')
add_argument('learning_rate', default=2e-3, dtype=float, msg='Learning rate.')
add_argument('value_learning_rate', default=2e-3, dtype=float, msg='Learning rate for value network.')
add_argument('keep_ratio', dtype=float, msg='Ratio of cognate pairs to keep.')
add_argument('test_keep_ratio', dtype=float, msg='Ratio of cognate pairs to keep for the test target language.')
add_argument('saved_model_path', dtype='path', msg='Path to the saved model.')
add_argument('evaluate_only', dtype=bool, default=False, msg='Flag to toggle evaluate-only mode.')
add_argument('share_src_tgt_abc', dtype=bool, default=False, msg='Flag to share the alphabets for source and target.')
add_argument('use_phono_features', dtype=bool, default=False, msg='Flag to use phonological features.')
add_argument('optim_cls', dtype=str, default='adam', choices=['sgd', 'adam'], msg='What optimizer to choose.')
add_argument('separate_value', dtype=bool, default=True,
             msg='Flag to use a separate model for value network. Used in RL.')
add_argument('max_rollout_length', default=10, dtype=int, msg='Maximum length of rollout')
add_argument('segments_dump_path', dtype='path', msg='Path to the processed Phoible pickle file.')
add_argument('use_max_value', dtype=bool, default=False, msg='Flag to use max mode in MCTS.')
add_argument('use_conditional', dtype=bool, default=True, msg='Flag to use conditional rules.')
add_argument('use_pruning', dtype=bool, default=True, msg='Flag to use pruning.')
add_argument('dist_threshold', dtype=float, default=0.0, msg='Distance threshold for pruning.')
add_argument('site_threshold', dtype=int, default=1, msg='Site threshold for pruning.')
add_argument('mcts_verbose_level', dtype=int, default=0, msg="Verbose level for debugging MCTS.")
add_argument('mcts_log_to_file', dtype=bool, default=False, msg="Flag to log to file for debugging MCTS.")
add_argument('add_noise', dtype=bool, default=False, msg="Flag to add noise to rewards.")
add_argument('use_num_misaligned', dtype=bool, default=False,
             msg="Flag to use number of misaligned characters instead of edit distance as heuristic.")
add_argument('use_alignment', dtype=bool, default=False, msg="Flag to use alignment to compute heuristics.")
add_argument('use_aligned_repr', dtype=bool, default=False,
             msg="Flag to use alignment to learned aligned representations.")

add_condition('use_phono_features', True, 'share_src_tgt_abc', True)
add_condition('use_rl', True, 'share_src_tgt_abc', True)
add_condition('use_mcts', True, 'use_rl', True)


class OnePairManager:
    """A manager for sound law induction on one src-tgt pair."""

    def __init__(self):
        # Prepare cognate registry.
        cr = CognateRegistry()
        cr.add_pair(g.data_path, g.src_lang, g.tgt_lang)

        # Prepare alphabets now.
        if g.use_mcts:
            with open(g.segments_dump_path, 'rb') as fin:
                segments_dump = pickle.load(fin)
            self.src_abc = self.tgt_abc = cr.prepare_alphabet(g.src_lang, g.tgt_lang, segments_dump=segments_dump)
            VocabState.abc = self.tgt_abc
        elif g.share_src_tgt_abc:
            self.src_abc = cr.prepare_alphabet(g.src_lang, g.tgt_lang)
            self.tgt_abc = self.src_abc
        else:
            self.src_abc = cr.prepare_alphabet(g.src_lang)
            self.tgt_abc = cr.prepare_alphabet(g.tgt_lang)

        # Prepare data loaders with different splits.
        self.dl_reg = DataLoaderRegistry()

        def create_setting(name: str, split: Split, for_training: bool,
                           keep_ratio: Optional[float] = None,
                           tgt_sot: bool = False) -> Setting:
            # If using RL, we append SOT's on the target side.
            return Setting(name, 'one_pair', split,
                           g.src_lang, g.tgt_lang, for_training,
                           keep_ratio=keep_ratio,
                           tgt_sot=tgt_sot)

        # Register all settings and data loaders.
        settings: List[Setting] = list()
        # For RL, we only need one data loader.
        if g.use_rl:
            # NOTE(j_luo) `for_training` is set to False since weighted sampling is not needed. SOT's are added on the target side.
            rl_setting = create_setting('rl', Split('all'), False, tgt_sot=True)
            settings.append(rl_setting)
        elif g.input_format == 'wikt':
            train_setting = create_setting('train', Split('train'), True,
                                           keep_ratio=g.keep_ratio)
            train_e_setting = create_setting('train_e', Split('train'), False,
                                             keep_ratio=g.keep_ratio)  # For evaluation.
            dev_setting = create_setting('dev', Split('dev'), False)
            settings.extend([train_setting, train_e_setting, dev_setting])
        else:
            for fold in range(5):
                dev_fold = fold + 1
                train_folds = list(range(1, 6))
                train_folds.remove(dev_fold)

                train_setting = create_setting(
                    f'train@{fold}', Split('train', train_folds), True, keep_ratio=g.keep_ratio)
                train_e_setting = create_setting(
                    f'train@{fold}_e', Split('train', train_folds), False,
                    keep_ratio=g.keep_ratio)
                dev_setting = create_setting(f'dev@{fold}', Split('dev', [dev_fold]), False)
                settings.extend([train_setting, train_e_setting, dev_setting])
        if not g.use_rl:
            test_setting = create_setting('test', Split('test'), False)
            settings.append(test_setting)

        for setting in settings:
            self.dl_reg.register_data_loader(setting, cr)

        # Get the RL environment if needed.
        if g.use_rl:
            dl = self.dl_reg.get_loaders_by_name('rl')
            # set_end_words(dl.end_state)
            src_seqs = dl.entire_batch.src_seqs
            s_arr = np.ascontiguousarray(src_seqs.ids.t().cpu().numpy()).astype('uint16')
            s_lengths = np.ascontiguousarray(src_seqs.lengths.t().cpu().numpy())
            tgt_seqs = dl.entire_batch.tgt_seqs
            t_arr = np.ascontiguousarray(tgt_seqs.ids.t().cpu().numpy()).astype('uint16')
            t_lengths = np.ascontiguousarray(tgt_seqs.lengths.t().cpu().numpy())
            env_opt = PyEnvOpt(s_arr, s_lengths, t_arr, t_lengths, g.final_reward, g.step_penalty)
            as_opt = PyActionSpaceOpt(NULL_ID, EMP_ID, SOT_ID, EOT_ID, ANY_ID, ANY_S_ID,
                                      ANY_UNS_ID, self.tgt_abc['j'], self.tgt_abc['w'], g.site_threshold, g.dist_threshold)
            ws_opt = PyWordSpaceOpt(self.tgt_abc.dist_mat, 0.5,
                                    g.use_alignment,
                                    self.tgt_abc.is_vowel,
                                    self.tgt_abc.unit_stress,
                                    self.tgt_abc.unit2base,
                                    self.tgt_abc.unit2stressed,
                                    self.tgt_abc.unit2unstressed)
            self.env = SoundChangeEnv(env_opt, as_opt, ws_opt, abc=self.tgt_abc)

    def run(self):
        phono_feat_mat = special_ids = None
        if g.use_phono_features:
            phono_feat_mat = get_tensor(self.src_abc.pfm)
            special_ids = get_tensor(self.src_abc.special_ids)

        metric_writer = MetricWriter(g.log_dir, flush_secs=5)

        def get_model(dl=None):
            phono_kwargs = {
                'phono_feat_mat': phono_feat_mat,
                'special_ids': special_ids
            }
            if g.use_rl:
                end_state = self.env.end
                agent_cls = VanillaPolicyGradient if g.agent == 'vpg' else A2C
                model = agent_cls(len(self.tgt_abc), self.env, end_state, **phono_kwargs)
            else:
                model = OnePairModel(len(self.src_abc), len(self.tgt_abc), **phono_kwargs)
            if g.saved_model_path is not None:
                model.load_state_dict(torch.load(g.saved_model_path, map_location=torch.device('cpu')))
                logging.imp(f'Loaded from {g.saved_model_path}.')
            if has_gpus():
                model.cuda()
            logging.info(model)
            return model

        def get_trainer(model, train_name, evaluator, metric_writer, **kwargs):
            if g.use_rl:
                # if g.use_mcts:
                trainer_cls = MctsTrainer
                # else:
                #     trainer_cls = PolicyGradientTrainer
            else:
                trainer_cls = Trainer
            trainer = trainer_cls(model, [self.dl_reg.get_setting_by_name(train_name)],
                                  [1.0], 'step',
                                  stage_tnames=['step'],
                                  check_interval=g.check_interval,
                                  evaluator=evaluator,
                                  eval_interval=g.eval_interval,
                                  save_interval=g.save_interval,
                                  metric_writer=metric_writer,
                                  **kwargs)
            if g.saved_model_path is None:
                # trainer.init_params('uniform', -0.1, 0.1)
                trainer.init_params('xavier_uniform')
            optim_cls = Adam if g.optim_cls == 'adam' else SGD
            optim_kwargs = dict()
            if optim_cls == SGD:
                optim_kwargs['momentum'] = 0.9
            if not g.use_rl or g.use_mcts or (g.agent == 'a2c' and g.value_steps == 0):
                trainer.set_optimizer(optim_cls, lr=g.learning_rate, weight_decay=g.weight_decay, **optim_kwargs)
            else:
                trainer.set_optimizer(optim_cls, name='policy', mod=model.policy_net,
                                      lr=g.learning_rate, **optim_kwargs)  # , weight_decay=1e-4)
                if g.agent == 'a2c':
                    trainer.set_optimizer(optim_cls, name='value', mod=model.value_net,
                                          lr=g.value_learning_rate, **optim_kwargs)  # , weight_decay=1e-4)
            return trainer

        def run_once(train_name, dev_name, test_name):
            train_e_dl = self.dl_reg[f'{train_name}_e']
            dev_dl = self.dl_reg[dev_name]
            test_dl = self.dl_reg[test_name]

            model = get_model()

            evaluator = Evaluator(model, {train_name: train_e_dl, dev_name: dev_dl, test_name: test_dl},
                                  self.tgt_abc,
                                  metric_writer=metric_writer)

            if g.evaluate_only:
                # TODO(j_luo) load global_step from saved model.
                evaluator.evaluate('evaluate_only', 0)
            else:
                trainer = get_trainer(model, train_name, evaluator, metric_writer)
                trainer.train(self.dl_reg)

        if g.use_rl:
            dl = self.dl_reg.get_loaders_by_name('rl')
            model = get_model(dl=dl)
            # if g.use_mcts:
            mcts_opt = PyMctsOpt(g.puct_c, g.game_count, g.virtual_loss, g.num_workers,
                                 g.heur_c, g.add_noise, g.use_num_misaligned)
            mcts = Mcts(self.env, mcts_opt, agent=model)
            # mcts.set_logging_options(g.mcts_verbose_level, g.mcts_log_to_file)
            if g.evaluate_only:
                tr = mcts.collect_episodes(mcts.env.start, num_episodes=1, is_eval=True)[0]
                # tr.save(g.log_dir)
                return
            else:
                trainer = get_trainer(model, 'rl', None, metric_writer, mcts=mcts)
            # else:
            #     collector = TrajectoryCollector(g.batch_size,
            #                                     max_rollout_length=g.max_rollout_length,
            #                                     truncate_last=True)
            #     trainer = get_trainer(model, 'rl', None, None, env=self.env, collector=collector)
            trainer.train(self.dl_reg)
        elif g.input_format == 'wikt':
            run_once('train', 'dev', 'test')
        else:
            for fold in range(5):
                logging.imp(f'Cross-validation, fold number {fold}')
                run_once(f'train@{fold}', f'dev@{fold}', 'test')


class OneToManyManager:
    """The manager class for single-source-multiple-target scenarios."""

    add_argument('train_tgt_langs', dtype=str, nargs='+', msg='Target languages used for training.')

    @staticmethod
    def prepare_raw_data() -> Tuple[List[str], CognateRegistry, Alphabet, Alphabet]:
        """Prepare raw data, including the cognates and the alphabets."""
        # Prepare cognate registry first.
        cr = CognateRegistry()
        all_tgt = sorted([g.tgt_lang] + list(g.train_tgt_langs))
        for tgt in all_tgt:
            cr.add_pair(g.data_path, g.src_lang, tgt)

        # Get alphabets. Note that the target alphabet is based on the union of all target languages, i.e., a shared alphabet for all.
        if g.share_src_tgt_abc:
            src_abc = cr.prepare_alphabet(*(all_tgt + [g.src_lang]))
            tgt_abc = src_abc
        else:
            src_abc = cr.prepare_alphabet(g.src_lang)
            tgt_abc = cr.prepare_alphabet(*all_tgt)

        return all_tgt, cr, src_abc, tgt_abc

    def __init__(self):
        all_tgt, self.cog_reg, self.src_abc, self.tgt_abc = self.prepare_raw_data()

        # Get stats for unseen units.
        stats = self.tgt_abc.stats
        _, test_tgt_path = get_paths(g.data_path, g.src_lang, g.tgt_lang)
        mask = (stats.sum() == stats.loc[test_tgt_path])
        unseen = mask[mask].index.tolist()
        total = len(stats.loc[test_tgt_path].dropna())
        logging.info(f'Unseen units ({len(unseen)}/{total}) for {g.tgt_lang} are: {unseen}.')

        # Get language-to-id mappings. Used only for the targets (i.e., decoder side).
        self.lang2id = lang2id = {tgt: i for i, tgt in enumerate(all_tgt)}

        # Get all data loaders.
        self.dl_reg = DataLoaderRegistry()

        def create_setting(name: str, tgt_lang: str, split: Split, for_training: bool,
                           keep_ratio: Optional[float] = None,
                           tgt_sot: bool = False) -> Setting:
            return Setting(name, 'one_pair', split,
                           g.src_lang, tgt_lang, for_training,
                           keep_ratio=keep_ratio,
                           tgt_sot=tgt_sot)

        test_setting = create_setting(f'test@{g.tgt_lang}', g.tgt_lang, Split('all'), False,
                                      keep_ratio=g.test_keep_ratio)
        settings: List[Setting] = [test_setting]

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
            train_e_setting = create_setting(f'train@{train_tgt_lang}_e',
                                             train_tgt_lang, train_split, False, keep_ratio=g.keep_ratio)
            dev_setting = create_setting(f'dev@{train_tgt_lang}', train_tgt_lang, dev_split, False)
            test_setting = create_setting(f'test@{train_tgt_lang}', train_tgt_lang, Split('test'), False)

            settings.extend([train_setting, train_e_setting, dev_setting, test_setting])
        for setting in settings:
            self.dl_reg.register_data_loader(setting, self.cog_reg, lang2id=lang2id)

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
                # self.trainer.init_params('uniform', -0.1, 0.1)
                self.trainer.init_params('xavier_uniform')
            optim_cls = Adam if g.optim_cls == 'adam' else SGD
            self.trainer.set_optimizer(optim_cls, lr=g.learning_rate)

    def run(self):
        if g.evaluate_only:
            # TODO(j_luo) load global_step from saved model.
            self.evaluator.evaluate('evaluate_only', 0)
        else:
            self.trainer.train(self.dl_reg)

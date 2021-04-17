import json
import math
import pickle
import re
import subprocess
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from itertools import product
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import altair as alt
import numpy as np
import pandas as pd
import psutil
import streamlit as st
import torch
from altair.vegalite.v4.schema.channels import Key
from sklearn.metrics import auc
from sound_law.utils import read_matching_score


class SwarmTransformer:
    """This is taken from https://github.com/mwaskom/seaborn/blob/master/seaborn/categorical.py."""

    def __call__(self, points):
        orig_xy = np.concatenate([points, np.full_like(points[:, 0], 0.01).reshape(-1, 1)], axis=-1)  # type:ignore
        sorter = np.argsort(orig_xy[:, 1])  # type: ignore
        orig_xyr = orig_xy[sorter]
        swarm = self.beeswarm(orig_xyr)
        new_xy = np.empty_like(points)
        new_xy[sorter] = swarm[:, :2]  # type: ignore
        return new_xy

    def beeswarm(self, orig_xyr):
        """Adjust x position of points to avoid overlaps."""
        # In this method, `x` is always the categorical axis
        # Center of the swarm, in point coordinates
        midline = orig_xyr[0, 0]

        # Start the swarm with the first point
        swarm = np.atleast_2d(orig_xyr[0])

        # Loop over the remaining points
        for xyr_i in orig_xyr[1:]:

            # Find the points in the swarm that could possibly
            # overlap with the point we are currently placing
            neighbors = self.could_overlap(xyr_i, swarm)

            # Find positions that would be valid individually
            # with respect to each of the swarm neighbors
            candidates = self.position_candidates(xyr_i, neighbors)

            # Sort candidates by their centrality
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]

            # Find the first candidate that does not overlap any neighbors
            new_xyr_i = self.first_non_overlapping_candidate(candidates, neighbors)

            # Place it into the swarm
            swarm = np.vstack([swarm, new_xyr_i])

        return swarm

    def could_overlap(self, xyr_i, swarm):
        """Return a list of all swarm points that could overlap with target."""
        # Because we work backwards through the swarm and can short-circuit,
        # the for-loop is faster than vectorization
        _, y_i, r_i = xyr_i
        neighbors = []
        for xyr_j in reversed(swarm):
            _, y_j, r_j = xyr_j
            if (y_i - y_j) < (r_i + r_j):
                neighbors.append(xyr_j)
            else:
                break
        return np.array(neighbors)[::-1]

    def position_candidates(self, xyr_i, neighbors):
        """Return a list of coordinates that might be valid by adjusting x."""
        candidates = [xyr_i]
        x_i, y_i, r_i = xyr_i
        left_first = True
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy ** 2, 0)) * 1.05  # type: ignore
            cl, cr = (x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i)
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    def first_non_overlapping_candidate(self, candidates, neighbors):
        """Find the first candidate that does not overlap with the swarm."""

        # If we have no neighbors, all candidates are good.
        if len(neighbors) == 0:
            return candidates[0]

        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]
        neighbors_r = neighbors[:, 2]

        for xyr_i in candidates:

            x_i, y_i, r_i = xyr_i

            dx = neighbors_x - x_i
            dy = neighbors_y - y_i
            sq_distances = np.square(dx) + np.square(dy)  # type: ignore

            sep_needed = np.square(neighbors_r + r_i)

            # Good candidate does not overlap any of neighbors which means that
            # squared distance between candidate and any of the neighbors has
            # to be at least square of the summed radii
            good_candidate = np.all(sq_distances >= sep_needed)

            if good_candidate:
                return xyr_i

        raise RuntimeError(
            "No non-overlapping candidates found. This should not happen."
        )


class TensorboardLaunchar:

    def __init__(self, port: int, *, selected_runs: Optional[Union[str, List[str]]] = None, saved_dir: Optional[str] = None):
        self.saved_dir = saved_dir
        if isinstance(selected_runs, str):
            selected_runs = [selected_runs]
        self.selected_runs = selected_runs
        self.port = port
        self.output: Optional[CompletedProcess] = None
        self._thread = None

    def launch(self):

        def run_cmd():
            if self.saved_dir:
                logdir_spec = self.saved_dir
            else:
                logdir_spec = ','.join([f'@{run}:{run}' for run in self.selected_runs])
            cmd = f'tensorboard --logdir_spec {logdir_spec} --host localhost --port {self.port} &'
            self.output = subprocess.run(cmd, capture_output=True, check=True, shell=True, text=True)

        self._thread = threading.Thread(target=run_cmd)
        self._thread.start()

    def is_successful(self) -> bool:
        time.sleep(3)
        return self.output is None


class HParamTuner:

    def __init__(self, base_cmd: str):
        self._base_cmd = base_cmd
        self._number_inputs = list()
        self._select_sliders = list()
        self._checkboxes = list()
        self._grid_variables = list()

    @property
    def hyperparameters(self) -> List[str]:
        ret = list()
        for ni in self._number_inputs:
            ret.append(ni[3][0])
        for ss in self._select_sliders:
            ret.append(ss[3][0])
        for cb in self._checkboxes:
            ret.append(cb[2][0])
        return ret

    def add_number_input(self, is_default_func, cmd_func, msg_func, *args, **kwargs):
        self._number_inputs.append((is_default_func, cmd_func, msg_func, args, kwargs))

    def add_select_slider(self, is_default_func, cmd_func, msg_func, *args, **kwargs):
        self._select_sliders.append((is_default_func, cmd_func, msg_func, args, kwargs))

    def add_checkbox(self, cmd_func, msg_func, *args, **kwargs):
        self._checkboxes.append((cmd_func, msg_func, args, kwargs))

    def render(self, grid_variables: List[str]) -> List[Tuple[str, str]]:

        def render_col(col, ui_type: str, col_args, col_kwargs):
            # For grid variable, render multiselect with all selected as default.
            label = col_args[0]
            if label in grid_variables:
                options = col_args[1]
                return col.multiselect(label, options, default=options)

            if ui_type == 'number_input':
                render_func = col.number_input
            elif ui_type == 'select_slider':
                render_func = col.select_slider
            else:
                render_func = col.checkbox
            return render_func(*col_args, **col_kwargs)

        def update_core(cmd_args: List[str], msg_args: List[str], value, is_default_func, cmd_func, msg_func):
            # For grid variables, we need to save it for later.
            if isinstance(value, list):
                self._grid_variables.append((value, is_default_func, cmd_func, msg_func))
                return

            if not is_default_func(value):
                cmd = cmd_func(value)
                msg = msg_func(value)
                if cmd:
                    cmd_args.append(cmd)
                if msg:
                    msg_args.append(msg)

        cmd_args = list()
        msg_args = list()
        # Render number_input.
        cols = list()
        for _ in range(0, len(self._number_inputs), 2):
            cols.extend(st.beta_columns(2))
        for col, (is_default_func, cmd_func, msg_func, args, kwargs) in zip(cols, self._number_inputs):
            value = render_col(col, 'number_input', args, kwargs)
            update_core(cmd_args, msg_args, value, is_default_func, cmd_func, msg_func)

        # Render select_slider.
        cols = list()
        for _ in range(0, len(self._select_sliders), 2):
            col1, _, col2, _ = st.beta_columns([2, 1, 2, 1])
            cols.extend([col1, col2])
        for col, (is_default_func, cmd_func, msg_func, args, kwargs) in zip(cols, self._select_sliders):
            value = render_col(col, 'select_slider', args, kwargs)
            update_core(cmd_args, msg_args, value, is_default_func, cmd_func, msg_func)

        # Render checkbox.
        cols = list()
        for _ in range(0, len(self._checkboxes), 3):
            cols.extend(st.beta_columns(3))
        for col, (cmd_func, msg_func, args, kwargs) in zip(cols, self._checkboxes):
            value = render_col(col, 'checkbox', args, kwargs)
            update_core(cmd_args, msg_args, value,
                        lambda x: x if kwargs.get('value', False) else not x, cmd_func, msg_func)

        if self._grid_variables:
            ret = list()
            for grid_cell in product(*[gv[0] for gv in self._grid_variables]):
                grid_cmd_args = cmd_args[:]
                grid_msg_args = msg_args[:]
                for i, value in enumerate(grid_cell):
                    is_default_func, cmd_func, msg_func = self._grid_variables[i][1:]
                    update_core(grid_cmd_args, grid_msg_args, value, is_default_func, cmd_func, msg_func)
                ret.append((self._base_cmd + ''.join(grid_cmd_args), '-'.join(grid_msg_args)))
            return ret
        else:
            return [(self._base_cmd + ''.join(cmd_args), '-'.join(msg_args))]


class JobScheduler:

    def __init__(self, job_path: str):
        self.job_path = Path(job_path)
        self._scheduler_thread = None

    def add_job_to_queue(self, job: str):
        with self.job_path.open('a', encoding='utf8') as fout:
            fout.write(job + '\n')

    def clear_job_queue(self):
        with self.job_path.open('w') as fout:
            pass

    def get_job_queue(self) -> List[str]:
        if self.job_path.exists():
            jobs = [job for job in self.job_path.open('r', encoding='utf8').read(-1).strip().split('\n') if job]
            if jobs:
                return jobs
        return list()

    def show_job_queue(self) -> List[str]:
        jobs = self.get_job_queue()
        if jobs:
            st.write(jobs)
        else:
            st.write('Job queue is empty.')
        return jobs

    def update_job_queue(self, job_queue: List[str]):
        with self.job_path.open('w', encoding='utf8') as fout:
            for job in job_queue:
                fout.write(job + '\n')

    def run(self):

        def run_impl():

            def get_spare_gpu() -> int:
                gpu_stats_output = subprocess.run('gpustat --json', capture_output=True,
                                                  check=True, text=True, shell=True)
                gpu_stats_json = json.loads(gpu_stats_output.stdout)
                idx2n = dict()
                for gpu_info in gpu_stats_json['gpus']:
                    gpu_idx = gpu_info['index']
                    if gpu_idx < 1:
                        continue
                    n_processes = len(gpu_info['processes'])
                    idx2n[gpu_idx] = n_processes
                idx, n_processes = sorted(idx2n.items(), key=lambda item: item[1])[0]
                if n_processes < 1:
                    return idx
                return -1

            def run_job(job: str):
                output = subprocess.run(job, capture_output=True, check=True, shell=True, text=True)

            def run_on_spare_gpu(job: str):
                while True:
                    time.sleep(10)
                    gpu_idx = get_spare_gpu()
                    if gpu_idx != -1:
                        job = job + f' --gpu {gpu_idx}'
                        thread = threading.Thread(target=run_job, args=(job, ))
                        thread.start()
                        break

            while True:
                job_queue = self.get_job_queue()
                if not job_queue:
                    break

                job = job_queue.pop(0)
                run_on_spare_gpu(job)
                self.update_job_queue(job_queue)

            self._scheduler_thread = None

        if self._scheduler_thread is None:
            self._scheduler_thread = threading.Thread(target=run_impl)
            self._scheduler_thread.start()


@st.cache(hash_funcs={JobScheduler: id})
def get_job_scheduler(job_queue_path: str) -> JobScheduler:
    return JobScheduler(job_queue_path)


if __name__ == "__main__":
    earliest_date = st.sidebar.date_input('Earliest date',
                                          value=date.fromisoformat('2021-03-28'),
                                          help='Earliest date to start looking for log files.')
    directory_choice = st.sidebar.radio('Which directory', ['log', 'saved'], index=0)
    with st.sidebar.beta_expander('Filter by'):
        regex = st.text_input('regex')

    with st.sidebar.beta_expander('Advanced'):
        log_dir = st.text_input('Log directory',
                                value='log',
                                help='Log directory with saved log files.')
        imp_dir = st.text_input('Important directory',
                                value='saved',
                                help='Directory to save important runs.')
        research_note_path = st.text_input('Research note',
                                           value='research_notes.tsv',
                                           help='File to store research notes.')
        job_queue_path = st.text_input('Job queue',
                                       value='job_queue.txt',
                                       help='File to store job queue.')

    log_dir = Path(log_dir)
    imp_dir = Path(imp_dir)
    js = get_job_scheduler(job_queue_path)

    # Show GPU stats if avaiable.
    if st.checkbox('show GPUs'):
        output = subprocess.run('command -v gpustat', capture_output=True, check=True, text=True, shell=True)
        if output.stdout:
            gpu_stats_output = subprocess.run('gpustat --json', capture_output=True, check=True, text=True, shell=True)
            gpu_stats_json = json.loads(gpu_stats_output.stdout)
            query_time = datetime.fromisoformat(gpu_stats_json['query_time'])
            st.write(query_time)
            gpu_df_records = list()
            for gpu_info in gpu_stats_json['gpus']:
                gpu_idx = gpu_info['index']
                gpu_mem_used = gpu_info['memory.used']
                gpu_mem_total = gpu_info['memory.total']
                gpu_df_records.append({'id': gpu_idx, 'mem_used': gpu_mem_used, 'mem_total': gpu_mem_total})
            gpu_df = pd.DataFrame(gpu_df_records)
            st.table(gpu_df)

    # Show CPU stats.
    if st.checkbox('show CPUs and RAM'):
        st.write(psutil.virtual_memory().available / 1024 / 1024 / 1024)
        st.write(psutil.cpu_count())
        st.write(psutil.cpu_percent())

    # Tensorboard stats.
    ports_to_use = list(range(8701, 8710))  # Use these ten ports.
    if st.beta_expander("Tensorboard"):
        output = subprocess.run('pgrep -u $(whoami) tensorboard', shell=True,
                                text=True, capture_output=True)

        def get_pid_info(pid: int) -> Tuple[int, str]:
            port = subprocess.run(
                f"ss -lp | grep pid={pid} | awk '{{print $5}}' | awk -F ':' '{{print $2}}'", shell=True, capture_output=True, text=True).stdout
            cmd = subprocess.run(f'ps -p {pid} -o args | tail -n 1', capture_output=True, text=True, shell=True).stdout
            return int(port), cmd

        tb_records = list()
        for pid in output.stdout.split():
            port, cmd = get_pid_info(int(pid))
            tb_records.append({'pid': pid, 'port': port, 'command': cmd})
            try:
                ports_to_use.remove(port)
            except ValueError:
                pass
        if tb_records:
            tb_info_df = pd.DataFrame(tb_records)
            st.table(tb_info_df)

            if st.button('Kill', help='Kill the tensorboard processes.'):
                for pid in tb_info_df['pid']:
                    subprocess.run(f'kill -9 {pid}', shell=True)
                st.write('tensorboard processes killed.')

    saved_runs = list()
    all_runs = st.checkbox('all_runs')
    latest_only = st.checkbox('latest_only')
    if directory_choice == 'log':
        latest_config = dict()
        for folder_with_date in log_dir.glob('*/'):
            try:
                folder_date = date.fromisoformat(folder_with_date.name)
            except ValueError:
                # Skip this if this doesn't conform to our directory naming convention.
                continue
            if folder_date >= earliest_date:
                for saved_run in folder_with_date.glob('*/*/'):
                    if not regex or re.search(regex, str(saved_run)):
                        if latest_only:
                            segs = str(saved_run).split('/')
                            config = segs[-2]
                            timestamp = str(folder_date) + ' ' + segs[-1]
                            if config not in latest_config or latest_config[config][0] < timestamp:
                                latest_config[config] = (timestamp, saved_run)
                        else:
                            saved_runs.append(saved_run)
        if latest_only:
            saved_runs = [v[1] for v in latest_config.values()]
    else:
        for saved_folder in imp_dir.glob('*/'):
            saved_runs.append(saved_folder)
    saved_runs = sorted(saved_runs, reverse=True)
    st.text(f'{len(saved_runs)} in total.')
    if not all_runs:
        selected_runs = st.multiselect('Saved run', saved_runs, help='Select saved runs to inspect.')
    else:
        selected_runs = saved_runs
    if selected_runs:
        # Op to launch tensorboard.
        if not ports_to_use:
            st.info('No usable ports left. Please kill more tensorboard processes.')
        if st.button('Launch tensorboard', key='launch_tensorboard'):
            port = ports_to_use[0]
            with st.spinner(f'Launching Tensorboard at port {port}...'):
                if all_runs:
                    launcher = TensorboardLaunchar(port, saved_dir=str(imp_dir))
                else:
                    launcher = TensorboardLaunchar(port, selected_runs=selected_runs)
                launcher.launch()
                if not launcher.is_successful():
                    st.error(launcher.output.stderr)

        # Op to mark important runs.
        if directory_choice == 'log':
            names = list()
            with st.beta_expander('Names'):
                for selected_run in selected_runs:
                    name_value = str(selected_run).split('/')[2]
                    name = st.text_input('name', help='The new name for this important run.',
                                         value=name_value)
                    names.append(name)

            col1, col2, col3 = st.beta_columns([1, 1, 5])
            col3_text = col3.empty()
            overwrite = col2.radio('overwrite', ['Y', 'N'], index=1, help='Overwrite the link.')
            if all(bool(name) for name in names):
                if col1.button('Mark'):
                    imp_dir.mkdir(parents=True, exist_ok=True)
                    for name, selected_run in zip(names, selected_runs):
                        link = imp_dir / name
                        if overwrite == 'Y':
                            link.unlink()
                        try:
                            link.symlink_to(selected_run.resolve(), target_is_directory=True)
                            col3_text.write('Marked.')
                        except FileExistsError:
                            col3_text.write('File already exists. Choose a different name.')
                            break

    # Job schedule.
    lang2code = {'Gothic': 'Got', 'Old Norse': 'Non', 'Old English': 'Ang'}
    with st.beta_expander('Job schedule'):
        lang = st.selectbox('language', ['Gothic', 'Old Norse', 'Old English'])
        lang2config = {k: 'OPRLPgmc' + v for k, v in lang2code.items()}
        config = lang2config[lang]
        base_cmd = f'python sound_law/main.py --config {config} --mcts_config SmallSims --save_interval 1'

        base_cmd = f'python sound_law/main.py --config {config} --mcts_config SmallSims --save_interval 1'

        ht = HParamTuner(base_cmd)
        ht.add_select_slider(lambda x: x == 2000,
                             lambda x: f' --num_mcts_sims {x} --expansion_batch_size {x // 20}',
                             lambda x: f'nms{x}',
                             'num_mcts_sims', [300, 500, 1000, 1500, 2000], value=2000)

        col1, col2 = st.beta_columns(2)
        ht.add_number_input(lambda x: x == 2,
                            lambda x: f' --site_threshold {x}',
                            lambda x: f'st{x}',
                            'site_threshold', value=2, min_value=1, max_value=5)
        ht.add_number_input(lambda x: x == 0.0,
                            lambda x: f' --dist_threshold {x}',
                            lambda x: f'dt{x}',
                            'dist_threshold', value=0.0)

        ht.add_checkbox(lambda x: ' --add_noise',
                        lambda x: 'noise',
                        'add_noise')
        ht.add_checkbox(lambda x: ' --use_num_misaligned',
                        lambda x: 'unm',
                        'use_num_misaligned')
        ht.add_checkbox(lambda x: ' --use_max_value',
                        lambda x: 'umv',
                        'use_max_value')

        ht.add_select_slider(lambda x: x == 'state',
                             lambda x: f' --repr_mode {x}',
                             lambda x: f'rm_{x}',
                             'repr_mode', ['state', 'word', 'char'], value='word')
        ht.add_select_slider(lambda x: x == 'max',
                             lambda x: f' --play_strategy {x}',
                             lambda x: f'ps_{x}',
                             'play_strategy', ['max', 'sample_ac', 'sample_mv'], value='sample_ac')
        ht.add_select_slider(lambda x: x == 50,
                             lambda x: f' --num_inner_steps {x}',
                             lambda x: f'nis{x}',
                             'num_inner_steps', [5, 10, 25, 50, 100, 200], value=50)
        ht.add_select_slider(lambda x: x == 1000,
                             lambda x: f' --replay_buffer_size {x}',
                             lambda x: f'rbs{x}',
                             'replay_buffer_size', [1000, 4000], value=1000)
        ht.add_select_slider(lambda x: x == 10,
                             lambda x: f' --num_episodes {x}',
                             lambda x: f'ne{x}',
                             'num_episodes', [10, 50], value=10)
        ht.add_select_slider(lambda x: x == 1,
                             lambda x: f' --puct_c {x}',
                             lambda x: f'puct{x}',
                             'puct_c', [1, 3, 5], value=1)
        ht.add_select_slider(lambda x: x == 1.0,
                             lambda x: f' --exponent {x}',
                             lambda x: f'exp{x}',
                             'exponent', [1, 2.5, 5, 7.5, 10], value=1.0)

        ht.add_checkbox(lambda x: ' --optim_cls sgd --learning_rate 0.01',
                        lambda x: 'sgd',
                        'SGD')
        ht.add_checkbox(lambda x: ' --improved_player_only',
                        lambda x: 'ipo',
                        'improved_player_only')

        decay2str = {
            1e-3: 'm3',
            1e-4: 'm4',
            3e-4: '3m4'
        }
        ht.add_select_slider(lambda x: x == 0.0,
                             lambda x: f' --weight_decay {x}',
                             lambda x: f'wd_{decay2str[x]}',
                             'weight_decay', [0.0, 1e-4, 3e-4, 1e-3], value=0.0)

        ht.add_select_slider(lambda x: x == 1.0,
                             lambda x: f' --heur_c {x}',
                             lambda x: f'heur{x}',
                             'heur_c', [0.0, 1.0, 5.0], value=1.0)

        grid_variables = st.multiselect('grid variable',
                                        ['num_mcts_sims', 'num_inner_steps', 'weight_decay', 'puct_c', 'exponent', 'play_strategy', 'num_episodes', 'heur_c',
                                         'repr_mode'])

        cmd_msg_pairs = ht.render(grid_variables)
        show_all_cmds = len(cmd_msg_pairs) <= 10
        st.text(f'There are {len(cmd_msg_pairs)} jobs in total.')

        jobs = list()
        for i, (cmd, default_msg) in enumerate(cmd_msg_pairs):
            if show_all_cmds:
                col1, col2 = st.beta_columns(2)
                msg = col1.text_input('message', help='The message to append to the run name.',
                                      value=default_msg)
                if msg:
                    cmd += f' --message {msg}'

                override_log_dir = col2.text_input('log_dir',
                                                   help='The actual log directory (overriding the default) to save everything.',
                                                   key=f'override_log_dir{i}')

                if override_log_dir:
                    cmd += f' --log_dir {override_log_dir}'

                # gpu_id = col2.number_input('GPU', value=i % 4, min_value=-1, max_value=3, key=f'gpu_job_{i}')
                # if gpu_id > -1:
                #     cmd += f' --gpu {gpu_id}'

                # Pretty-print command.
                st.markdown("Command to run:\n```\n" + cmd.replace(' --', '  \n  --') + "\n```")

                # Add a new job.
                if st.button('Add job', key=f'add_job_{i}'):
                    js.add_job_to_queue(cmd)
            else:
                cmd += f' --message {default_msg}'
            jobs.append(cmd)

        if st.button('Add all jobs', key='add_all_jobs'):
            for cmd in jobs:
                js.add_job_to_queue(cmd)

        job_queue = js.show_job_queue()
        col1, col2 = st.beta_columns(2)
        if job_queue:
            if col2.button('Clear job queue', key='clear_job_queue'):
                js.clear_job_queue()
        if col1.button('Run job queue', key='run_job_queue'):
            js.run()

    # Take research notes.
    with st.beta_expander('Research note'):
        rnp = Path(research_note_path)
        if not rnp.exists():
            rn_df = pd.DataFrame([], columns=['Timestamp', 'Run', 'Note'])
        else:
            rn_df = pd.read_csv(research_note_path, sep='\t')
        no_run = st.checkbox('no_run')
        path = ''
        if not no_run:
            path = st.selectbox('Which run', saved_runs)
        note = st.text_area('Research note')
        if st.button('Add research note') and note:
            rn_df = rn_df.append({'Timestamp': datetime.now().strftime('%y-%m-%e %H:%M:%S'),
                                  'Run': path,
                                  'Note': note},
                                 ignore_index=True)
            rn_df.to_csv(research_note_path, sep='\t', index=False)
        st.write(rn_df)

    if selected_runs:
        with st.beta_expander('Analysis'):
            if st.checkbox('show distance'):
                pbar = st.progress(0.0)
                record_dfs = list()
                ht_hparams = ht.hyperparameters
                # HACK(j_luo) SGD uses different param names.
                ht_hparams.remove('SGD')
                ht_hparams.extend(['optim_cls', 'learning_rate', 'tgt_lang'])

                for run_id, selected_run in enumerate(selected_runs, 1):
                    # Get all hparams.
                    hparams_path = Path(selected_run) / 'hparams.pth'
                    state_dict = torch.load(str(hparams_path))
                    hparams = {hp: state_dict[hp].value for hp in ht_hparams}

                    # Extract events and incorporate hparams into the records as well.
                    match_record = load_event(selected_run)
                    match_record = match_record.assign(**hparams)
                    record_dfs.append(match_record)
                    pbar.progress(run_id / len(selected_runs))
                record_df: pd.DataFrame = pd.concat(record_dfs, ignore_index=True)  # type: ignore
                scores_df = record_df[record_df['tag'] == 'best_score'][['run', 'step', 'value'] + ht_hparams]
                cols_to_show = ['value'] + ht_hparams

                best_score_df = scores_df.sort_values(by=['run', 'step']).pivot_table(
                    index='run', values=cols_to_show, aggfunc={col: 'last' for col in cols_to_show}).reset_index()  # type: ignore

                col1, col2 = st.beta_columns(2)
                heur = col1.radio('heur', [None, '+', '-'])
                big = col2.radio('big', [None, '+', '-'])
                hparam_mask = pd.Series([True] * len(best_score_df))
                if heur:
                    hparam_mask &= best_score_df['heur_c'] == (1.0 if heur == '+' else 0.0)
                if big:
                    hparam_mask &= best_score_df['num_mcts_sims'] == (2000 if big == '+' else 300)
                hparam_mask &= (best_score_df['puct_c'] == 1.0) & (best_score_df['repr_mode'] == 'state')

                to_inspect = best_score_df[hparam_mask]
                st.write(to_inspect)

                # best_score_df.to_csv('best_score.tsv', sep='\t', index=False)

                row_var = 'tgt_lang'
                col_var = 'play_strategy'
                group_var = 'num_mcts_sims'
                bar_var = 'heur_c'
                style_var = 'use_max_value'

                def sort_values(values):
                    if 'got' in values:
                        return ['got', 'non', 'ang']
                    else:
                        return sorted(values)

                row_values = sort_values(set(to_inspect[row_var]))  # type: ignore
                col_values = sort_values(set(to_inspect[col_var]))  # type: ignore
                group_values = sort_values(set(to_inspect[group_var]))  # type: ignore

                swarm_tr = SwarmTransformer()

                def get_bar_chart(bar_df, min_value: float, max_value: float):
                    charts = list()
                    for i, v in enumerate(sorted(set(bar_df[bar_var]))):
                        points_subset_df = bar_df[bar_df[bar_var] == v].assign(x=i)
                        points_subset = points_subset_df[['x', 'value']].values
                        points_subset = swarm_tr(points_subset)
                        points_df = pd.DataFrame(points_subset, columns=['x', 'y'])
                        points_df = pd.concat(
                            [points_df, points_subset_df[[bar_var, style_var]].reset_index(drop=True)], axis=1)
                        charts.append(alt.Chart(points_df, width=40).mark_point(filled=True, size=50).encode(
                            x='x:Q',
                            y=alt.Y('y:Q',
                                    scale=alt.Scale(domain=(min_value, max_value), nice=False)),
                            color=f'{bar_var}:N',
                            shape=f'{style_var}:N'))
                    return alt.layer(*charts)

                def get_group_chart(grid_df, min_value: float, max_value: float, title: str = ''):
                    rcharts = list()
                    for gv in group_values:
                        bar_df = grid_df[grid_df[group_var] == gv]
                        rcharts.append(get_bar_chart(bar_df, min_value, max_value))
                    return alt.hconcat(*rcharts,
                                       title=alt.TitleParams(title,
                                                             anchor='middle',
                                                             align='center',
                                                             orient='top'))

                chart = alt.vconcat()

                min_unit = 0.05
                for r, rv in enumerate(row_values):
                    rcharts = list()
                    row_df = to_inspect[to_inspect[row_var] == rv]
                    min_value = math.floor(row_df['value'].min() / min_unit) * min_unit  # type: ignore
                    max_value = math.ceil(row_df['value'].max() / min_unit) * min_unit  # type: ignore
                    for cv in col_values:
                        grid_df = to_inspect[(to_inspect[row_var] == rv) & (to_inspect[col_var] == cv)]
                        # Only add title at the top row.
                        title = f'{col_var} = {cv}' if r == 0 else ''
                        rcharts.append(get_group_chart(grid_df, min_value, max_value, title=title))
                    chart &= alt.hconcat(*rcharts,
                                         title=alt.TitleParams(f'{row_var} = {rv}',
                                                               anchor='middle',
                                                               align='center',
                                                               orient='right'))
                chart.configure_view(stroke=None)

                st.write(chart)

                average_col = st.selectbox('average_col',
                                           [None, 'play_strategy', 'use_max_value', 'heur_c', 'num_mcts_sims'])
                if average_col:
                    st.write(to_inspect.pivot_table(index=average_col, values='value', aggfunc='mean'))
            if st.checkbox('show matching'):
                # Get all matching scores.
                path = st.selectbox('which run', [None] + selected_runs)
                if path is not None:
                    auc_score, match_df = read_matching_metrics(path)

                    chart = alt.Chart(match_df).mark_rect().encode(
                        x='max_power_set_size:O',
                        y='k_matches:O',
                        color='score:Q',
                        tooltip=[
                            alt.Tooltip('max_power_set_size:O', title='max_power_set_size'),
                            alt.Tooltip('k_matches:O', title='k_matches'),
                            alt.Tooltip('score:Q', title='score'),
                        ]
                    ).facet(column='match_proportion')
                    st.write(chart)
                pbar = st.progress(0.0)
                dfs = list()
                all_match_results = list()
                for i, run in enumerate(selected_runs, 1):
                    auc_score, match_df = read_matching_metrics(run)
                    lang = re.search(r'OPRLPgmc(\w\w\w)', str(run)).group(1).lower()
                    match_df = match_df.assign(run=run, lang=lang)
                    event_df = load_event(run)
                    dfs.append(match_df)
                    all_match_results.append((auc_score, match_df, event_df))
                    pbar.progress(i / len(selected_runs))
                with open('matching_results.pkl', 'wb') as fout:
                    pickle.dump(all_match_results, fout)
                match_df = pd.concat(dfs, ignore_index=True)
                st.write(match_df)
                match_df.to_csv('match_heatmap.tsv', sep='\t', index=False)

                # Use more correlation data by adding truncated paths.
                corr_data = list()
                lang2length = {'got': 20, 'non': 40, 'ang': 60}
                for i, run in enumerate(selected_runs, 1):
                    lang = re.search(r'OPRLPgmc(\w\w\w)', str(run)).group(1).lower()
                    best_run = int((Path(run) / 'best_run').open('r').read(-1).strip())
                    length = lang2length[lang]
                    records = list()
                    score_path = f'{run}/eval/{best_run}.path.scores'
                    with open(score_path) as fin:
                        truncated_dists = list()
                        for line in fin:
                            truncated_dists.append(float(line.strip()))
                    start_dist = truncated_dists[0]
                    # last_record = None
                    for l in range(5, length + 1, 5):
                        if l >= len(truncated_dists):
                            break
                        match_record = {'truncate_length': l,
                                        'best_score': 1.0 - truncated_dists[l] / start_dist, 'lang': lang, 'run': run}
                        scores = [1.0]
                        for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
                            match_score = read_matching_score(f'{run}/eval/{m}-100-10-{l}.pkl')
                            scores.append(match_score)
                            match_record[f'match_{m}'] = match_score
                        assert all(score > -1 for score in scores)
                        auc_score = auc([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], scores)
                        match_record['auc_score'] = auc_score
                        # last_record = record
                        corr_data.append(match_record)
                    # corr_data.append(last_record)
                corr_df = pd.DataFrame(corr_data)
                # corr_df.to_csv('corr_df.tsv', sep='\t', index=False)
                st.write(corr_df)

                chart = alt.Chart(corr_df).mark_point().encode(
                    x='best_score:Q',
                    y='auc_score:Q',
                    tooltip=[
                        alt.Tooltip('best_score:Q', title='best_score'),
                        alt.Tooltip('auc_score:Q', title='auc_score'),
                    ]
                ).interactive().facet(row='lang')
                st.write(chart)
            if st.checkbox('show synthetic results'):
                stats_dfs = list()
                record_dfs = list()
                match_records = list()
                for run in selected_runs:
                    record_df = load_event(run).assign(run=run)
                    rand_idx = re.search(r'rand(\d+)', str(run)).group(1)
                    # rand_idx = re.search(r'rand-regress(\d+)', str(run)).group(1)
                    # rand_idx = re.search(r'rand-merger(\d+)', str(run)).group(1)
                    # stats_df = load_stats(f'data/wikt/pgmc-rand-regress{rand_idx}').assign(run=run)
                    # stats_df = load_stats(f'data/wikt/pgmc-rand-merger{rand_idx}').assign(run=run)
                    stats_df = load_stats(f'data/wikt/pgmc-rand{rand_idx}').assign(run=run)
                    stats_dfs.append(stats_df)
                    record_dfs.append(record_df)
                    scores = [1.0]
                    match_record = {'run': run}
                    for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
                        match_score = read_matching_score(f'{run}/eval/{m}-100-10.pkl')
                        scores.append(match_score)
                        match_record[f'match_{m}'] = match_score
                    assert all(score > -1 for score in scores)
                    auc_score = auc([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], scores)
                    match_record['auc_score'] = auc_score
                    match_records.append(match_record)
                stats_df = pd.concat(stats_dfs, ignore_index=True)
                record_df = pd.concat(record_dfs, ignore_index=True)  # type:ignore
                match_df = pd.DataFrame(match_records)
                best_score_df = record_df[record_df['tag'] == 'best_score'][['run', 'value', 'step']]
                best_score_df = best_score_df.sort_values(by='run').pivot_table(
                    index='run', values='value', aggfunc='max').reset_index()
                stats_df = stats_df.merge(best_score_df, left_on='run', right_on='run')
                stats_df = stats_df.merge(match_df, left_on='run', right_on='run')
                stats_df = stats_df.rename(columns={'value': 'best_score'})
                st.write(stats_df)

                # st.bar_chart(stats_df['n_regress'].value_counts())
                st.bar_chart(stats_df['n_merger'].value_counts())
                # st.bar_chart(stats_df['n_split'].value_counts())
                # st.bar_chart(stats_df['n_loss'].value_counts())
                # st.bar_chart(stats_df['n_irreg'].value_counts())

                # merger_box_plot = alt.Chart(stats_df).mark_boxplot().encode(
                #     x='n_merger:O',
                #     y='best_score:Q'
                # )

                # split_box_plot = alt.Chart(stats_df).mark_boxplot().encode(
                #     x='n_split:O',
                #     y='best_score:Q'
                # )

                # loss_box_plot = alt.Chart(stats_df).mark_boxplot().encode(
                #     x='n_loss:O',
                #     y='best_score:Q'
                # )

                box_plot = alt.Chart(stats_df).mark_boxplot().encode(
                    x='n_irreg:O',
                    # x='n_merger:O',
                    # x='n_regress:O',
                    # y='best_score:Q'
                    y='auc_score:Q'
                )
                st.write(box_plot)
                stats_df.to_csv('syn_regress.tsv', sep='\t', index=False)

                # col1, col2, col3 = st.beta_columns(3)
                # col1.write(merger_box_plot)
                # col2.write(split_box_plot)
                # col3.write(loss_box_plot)

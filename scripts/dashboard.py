import json
import re
import subprocess
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import streamlit as st


class TensorboardLaunchar:

    def __init__(self, port: int, *, selected_runs: Optional[Union[str, List[str]]] = None, saved_dir: Optional[str] = None):
        self.saved_dir = saved_dir
        if isinstance(selected_runs, str):
            selected_runs = [selected_runs]
        self.selected_runs = selected_runs
        self.port = port
        self.output = None
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


class HyperparameterTuner:

    def __init__(self, base_cmd: str):
        self._cmd = base_cmd
        self._msg_args = list()
        self._number_inputs = list()
        self._select_sliders = list()
        self._checkboxes = list()

    def add_number_input(self, is_default_func, cmd_func, msg_func, *args, **kwargs):
        self._number_inputs.append((is_default_func, cmd_func, msg_func, args, kwargs))

    def add_select_slider(self, is_default_func, cmd_func, msg_func, *args, **kwargs):
        self._select_sliders.append((is_default_func, cmd_func, msg_func, args, kwargs))

    def add_checkbox(self, cmd_func, msg_func, *args, **kwargs):
        self._checkboxes.append((cmd_func, msg_func, args, kwargs))

    def render(self):

        def render_col(col, ui_type: str, col_args, col_kwargs):
            if ui_type == 'number_input':
                render_func = col.number_input
            elif ui_type == 'select_slider':
                render_func = col.select_slider
            elif ui_type == 'checkbox':
                render_func = col.checkbox
            return render_func(*col_args, **col_kwargs)

        def update_core(value, is_default_func, cmd_func, msg_func):
            if not is_default_func(value):
                cmd = cmd_func(value)
                msg = msg_func(value)
                if cmd:
                    self._cmd += cmd
                if msg:
                    self._msg_args.append(msg)

        # Render number_input.
        cols = list()
        for _ in range(0, len(self._number_inputs), 2):
            cols.extend(st.beta_columns(2))
        for col, (is_default_func, cmd_func, msg_func, args, kwargs) in zip(cols, self._number_inputs):
            value = render_col(col, 'number_input', args, kwargs)
            update_core(value, is_default_func, cmd_func, msg_func)

        # Render select_slider.
        cols = list()
        for _ in range(0, len(self._select_sliders), 2):
            col1, _, col2, _ = st.beta_columns([2, 1, 2, 1])
            cols.extend([col1, col2])
        for col, (is_default_func, cmd_func, msg_func, args, kwargs) in zip(cols, self._select_sliders):
            value = render_col(col, 'select_slider', args, kwargs)
            update_core(value, is_default_func, cmd_func, msg_func)

        # Render checkbox.
        cols = list()
        for _ in range(0, len(self._checkboxes), 3):
            cols.extend(st.beta_columns(3))
        for col, (cmd_func, msg_func, args, kwargs) in zip(cols, self._checkboxes):
            value = render_col(col, 'checkbox', args, kwargs)
            update_core(value, lambda x: x if kwargs.get('value', False) else not x, cmd_func, msg_func)

    @property
    def default_msg(self) -> str:
        return '-'.join(self._msg_args)

    @property
    def cmd(self) -> str:
        return self._cmd


if __name__ == "__main__":
    # Show GPU stats if avaiable.
    with st.beta_expander('GPU'):
        to_refresh = st.button('Refresh')
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

    # Tensorboard stats.
    with st.beta_expander("Tensorboard"):
        output = subprocess.run('pgrep -u $(whoami) tensorboard', shell=True,
                                text=True, capture_output=True)

        ports_to_use = list(range(8701, 8710))  # Use these ten ports.

        def get_pid_info(pid: int) -> Tuple[int, str]:
            port = subprocess.run(
                f"ss -lp | grep pid={pid} | awk '{{print $5}}' | awk -F ':' '{{print $2}}'", shell=True, capture_output=True, text=True).stdout
            cmd = subprocess.run(f'ps -p {pid} -o args | tail -n 1', capture_output=True, text=True, shell=True).stdout
            return int(port), cmd

        tb_records = list()
        for pid in output.stdout.split():
            port, cmd = get_pid_info(pid)
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

    log_dir = Path(log_dir)
    imp_dir = Path(imp_dir)
    saved_runs = list()
    all_runs = False
    if directory_choice == 'log':
        for folder_with_date in log_dir.glob('*/'):
            folder_date = date.fromisoformat(folder_with_date.name)
            if folder_date >= earliest_date:
                for saved_run in folder_with_date.glob('*/*/'):
                    if not regex or re.search(regex, str(saved_run)):
                        saved_runs.append(saved_run)
    else:
        for saved_folder in imp_dir.glob('*/'):
            saved_runs.append(saved_folder)
        all_runs = st.checkbox('all_runs')
    saved_runs = sorted(saved_runs, reverse=True)

    selected_runs = st.multiselect('Saved run', saved_runs, help='Select saved runs to inspect.')
    if selected_runs or all_runs:
        # Op to launch tensorboard.
        if not ports_to_use:
            st.info('No usable ports left. Please kill more tensorboard processes.')
        if st.button('Launch tensorboard', key='launch_tensorboard'):
            port = ports_to_use[0]
            with st.spinner(f'Launching Tensorboard at port {port}...'):
                if all_runs:
                    launcher = TensorboardLaunchar(port, saved_dir=imp_dir)
                else:
                    launcher = TensorboardLaunchar(port, selected_runs=selected_runs)
                launcher.launch()
                if not launcher.is_successful():
                    st.error(launcher.output.stderr)
        # Op to mark an important run.
        if directory_choice == 'log' and len(selected_runs) == 1:
            selected_run = selected_runs[0]
            with st.beta_expander("Mark as important"):
                name = st.text_input('name', help='The new name for this important run.',
                                     value=str(selected_run).split('/')[2])
                col1, col2, col3 = st.beta_columns([1, 1, 5])
                overwrite = col2.radio('overwrite', ['Y', 'N'], index=1, help='Overwrite the link.')
                if col1.button('Mark') and name:
                    with st.spinner('Marking as important'):
                        imp_dir.mkdir(parents=True, exist_ok=True)
                        link = imp_dir / name
                        if overwrite == 'Y':
                            link.unlink()
                        try:
                            link.symlink_to(selected_run.resolve(), target_is_directory=True)
                            col3.write('Marked.')
                        except FileExistsError:
                            col3.write('File already exists. Choose a different name.')

    # Job creation.
    with st.beta_expander('Job creation'):
        lang = st.selectbox('language', ['Gothic', 'Old Norse', 'Old English'])
        lang2config = {'Gothic': 'Got', 'Old Norse': 'Non', 'Old English': 'Ang'}
        lang2config = {k: 'OPRLPgmc' + v for k, v in lang2config.items()}
        config = lang2config[lang]
        base_cmd = f'python sound_law/main.py --config {config} --mcts_config SmallSims --save_interval 1'

        ht = HyperparameterTuner(base_cmd)
        ht.add_number_input(lambda x: x == -1,
                            lambda x: f' --gpu {x}',
                            lambda x: None,
                            'GPU', value=-1, min_value=-1, max_value=3)

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
        ht.add_checkbox(lambda x: ' --use_aligned_repr',
                        lambda x: 'uar',
                        'used_aligned_repr')
        ht.add_checkbox(lambda x: ' --use_num_misaligned',
                        lambda x: 'unm',
                        'use_num_misaligned')
        ht.add_checkbox(lambda x: ' --use_max_value',
                        lambda x: 'umv',
                        'use_max_value')

        ht.add_select_slider(lambda x: x == 50,
                             lambda x: f' --num_inner_steps {x}',
                             lambda x: f'nis{x}',
                             'num_inner_steps', [5, 50], value=50)
        ht.add_select_slider(lambda x: x == 1000,
                             lambda x: f' --replay_buffer_size {x}',
                             lambda x: f'rbs{x}',
                             'replay_buffer_size', [1000, 4000], value=1000)
        ht.add_select_slider(lambda x: x == 10,
                             lambda x: f' --num_episodes {x}',
                             lambda x: f'ne{x}',
                             'num_episodes', [10, 50], value=10)

        ht.add_checkbox(lambda x: ' --optim_cls sgd --learning_rate 0.1',
                        lambda x: 'sgd',
                        'SGD')

        ht.add_select_slider(lambda x: x == 0.0,
                             lambda x: f' --weight_decay {x}',
                             lambda x: f'wd_m{3 if x == 1e-3 else 4}',
                             'weight_decay', [0.0, 1e-4, 1e-3], value=0.0)

        ht.add_select_slider(lambda x: x == 1.0,
                             lambda x: f' --heur_c {x}',
                             lambda x: f'heur{x}',
                             'heur_c', [1.0, 5.0], value=1.0)

        ht.render()

        msg = st.text_input('message', help='The message to append to the run name.',
                            value=ht.default_msg)

        cmd = ht.cmd
        if msg:
            cmd += f' --message {msg}'

        # Pretty-print command.
        st.markdown("Command to run:\n```\n" + cmd.replace(' --', '  \n  --') + "\n```")

        # Launch a new job.
        if st.button('Launch job', key='launch_job'):
            with st.spinner(f'Launching job "{cmd}"'):

                def run_cmd():
                    output = subprocess.run(cmd, capture_output=True, check=True, shell=True, text=True)

                thread = threading.Thread(target=run_cmd)
                thread.start()
                time.sleep(3)

    # Take research notes.
    with st.beta_expander('Research note'):
        rnp = Path(research_note_path)
        if not rnp.exists():
            rn_df = pd.DataFrame([], columns=['Timestamp', 'Run', 'Note'])
        else:
            rn_df = pd.read_csv(research_note_path, sep='\t')
        run = st.selectbox('Which run', saved_runs)
        note = st.text_area('Research note')
        if st.button('Add research note') and note:
            rn_df = rn_df.append({'Timestamp': datetime.now().strftime('%y-%m-%e %H:%M:%S'),
                                  'Run': run,
                                  'Note': note},
                                 ignore_index=True)
            rn_df.to_csv(research_note_path, sep='\t', index=None)
        st.write(rn_df)

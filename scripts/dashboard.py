import json
import re
import subprocess
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st


class TensorboardLaunchar:

    def __init__(self, selected_run: str, port: int):
        self.selected_run = selected_run
        self.port = port
        self.output = None
        self._thread = None

    def launch(self):

        def run_cmd():
            cmd = f'tensorboard --logdir {self.selected_run} --host localhost --port {self.port} &'
            self.output = subprocess.run(cmd, capture_output=True, check=True, shell=True, text=True)

        self._thread = threading.Thread(target=run_cmd)
        self._thread.start()

    def is_successful(self) -> bool:
        time.sleep(3)
        return self.output is None


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
            st.write(gpu_df)

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

    log_dir = Path(log_dir)
    imp_dir = Path(imp_dir)
    saved_runs = list()
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
    saved_runs = sorted(saved_runs, reverse=True)

    selected_run = st.selectbox('Saved run', saved_runs, help='Select a saved run to inspect.')
    if selected_run:
        # Op to mark an important run.
        if directory_choice == 'log':
            with st.beta_expander("Mark as important"):
                name = st.text_input('name', help='The new name for this important run.')
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

        # Tensorboard related ops.
        with st.beta_expander("Tensorboard"):
            port = st.number_input('Port', value=8701)
            col1, col2, _ = st.beta_columns([1, 1, 5])
            if col1.button('Launch', key='launch_tensorboard'):
                with st.spinner(f'Launching Tensorboard at port {port}...'):
                    launcher = TensorboardLaunchar(selected_run, port)
                    launcher.launch()
                    if not launcher.is_successful():
                        st.error(launcher.output.stderr)
            output = subprocess.run('pgrep -u $(whoami) tensorboard', shell=True,
                                    text=True, capture_output=True)
            if output.stdout and col2.button('Kill', help='Kill the tensorboard processes.'):
                for pid in output.stdout.split():
                    subprocess.run(f'kill -9 {pid}', shell=True)
                st.write('tensorboard processes killed.')

    # Job creation.
    with st.beta_expander('Job creation'):
        col1, col2 = st.beta_columns(2)

        lang = col1.selectbox('language', ['Gothic', 'Old Norse', 'Old English'])
        lang2config = {'Gothic': 'Got', 'Old Norse': 'Non', 'Old English': 'Ang'}
        lang2config = {k: 'OPRLPgmc' + v for k, v in lang2config.items()}
        config = lang2config[lang]

        gpu_id = col2.number_input('GPU', value=-1, min_value=-1, max_value=3)
        cmd = f'python sound_law/main.py --config {config} --mcts_config SmallSims --save_interval 1'
        if gpu_id > -1:
            cmd += f' --gpu {gpu_id}'

        num_mcts_sims = st.select_slider('num_mcts_sims', [300, 500, 1000, 1500, 2000], value=2000)
        expansion_batch_size = num_mcts_sims // 20
        if num_mcts_sims != 2000:
            cmd += f' --num_mcts_sim {num_mcts_sims}'
            cmd += f' --expansion_batch_size {expansion_batch_size}'

        col1, col2 = st.beta_columns(2)
        site_threshold = col1.number_input('site_threshold', value=2, min_value=1, max_value=5)
        dist_threshold = col2.number_input('dist_threshold', value=0.0)
        if site_threshold != 2:
            cmd += f' --site_threshold {site_threshold}'
        if dist_threshold != 0.0:
            cmd += f' --dist_threshold {dist_threshold}'

        if st.checkbox('add_noise'):
            cmd += ' --add_noise'

        msg = st.text_input('message', help='The message to append to the run name.')
        if msg:
            cmd += f' --message {msg}'

        # Pretty-print command.
        st.markdown("Command to run:\n```\n" + cmd.replace(' --', '  \n  --') + "\n```")

        # Launch a new job.
        if st.button('Launch', key='launch_job'):
            with st.spinner(f'Launching job "{cmd}"'):

                def run_cmd():
                    output = subprocess.run(cmd, capture_output=True, check=True, shell=True, text=True)

                thread = threading.Thread(target=run_cmd)
                thread.start()
                time.sleep(3)

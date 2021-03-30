import torch
import time
import subprocess
import threading
from datetime import date
from pathlib import Path

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
    earliest_date = st.sidebar.date_input('Earliest date',
                                          value=date.fromisoformat('2021-03-28'),
                                          help='Earliest date to start looking for log files.')
    if st.sidebar.checkbox('Select date', help='Select a date to look for log files.'):
        selected_date = st.sidebar.date_input('Selected date',
                                              min_value=earliest_date,
                                              help='Selected date to look for log files.')
    else:
        selected_date = None

    with st.sidebar.beta_expander('Advanced'):
        log_dir = st.text_input('Log directory',
                                value='log',
                                help='Log directory with saved log files.')
        imp_dir = st.text_input('Important directory',
                                value='saved',
                                help='Directory to save important runs.')

    log_dir = Path(log_dir)
    imp_dir = Path(imp_dir)
    pat = '*/' if selected_date is None else f'{selected_date}/'
    saved_runs = list()
    for folder_with_date in log_dir.glob(pat):
        folder_date = date.fromisoformat(folder_with_date.name)
        if folder_date >= earliest_date:
            for saved_run in folder_with_date.glob('*/*/'):
                saved_runs.append(saved_run)
    saved_runs = sorted(saved_runs, reverse=True)

    selected_run = st.selectbox('Saved run', saved_runs, help='Select a saved run to inspect.')
    if selected_run:
        # Op to mark an important run.
        with st.beta_expander("Mark as important"):
            name = st.text_input('name', help='The new name for this important run.')
            if st.button('Mark') and name:
                with st.spinner('Marking as important'):
                    imp_dir.mkdir(parents=True, exist_ok=True)
                    link = imp_dir / name
                    link.symlink_to(selected_run.resolve(), target_is_directory=True)

        # Tensorboard related ops.
        with st.beta_expander("Tensorboard"):
            port = st.number_input('Port', value=8701)
            if st.button('Launch', key='launch_tensorboard'):
                with st.spinner(f'Launching Tensorboard at port {port}...'):
                    launcher = TensorboardLaunchar(selected_run, port)
                    launcher.launch()
                    if not launcher.is_successful():
                        st.error(launcher.output.stderr)

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

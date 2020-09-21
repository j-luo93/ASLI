import sys
from multiprocessing import current_process

from dask.distributed import Client, Worker, WorkerPlugin, get_worker, progress
from dask_cuda import LocalCUDACluster

from dev_misc import Initiator, add_argument, g, parse_args, show_args
from dev_misc.arglib import disable_duplicate_check, test_with_arguments
from dev_misc.devlib.grid import make_grid
from sound_law.main import main, setup


def main_distr(cmd: str):
    worker = get_worker()
    sys.argv = [sys.argv[0]] + cmd.split()
    main(worker.initiator)


class WorkerSetup(WorkerPlugin):

    def setup(self, worker: Worker):
        worker.initiator = setup()


if __name__ == '__main__':
    add_argument('gpus', nargs='+', dtype=int, msg='GPUs to use.')
    add_argument('configs', nargs='+', dtype=str, msg='All configs to run.')
    add_argument('grid_file_path', dtype=str, msg='Path to a grid file.')
    parse_args()

    if g.grid_file_path:
        cmdl = make_grid(g.grid_file_path)
    else:
        cmdl = [f'--config {config}' for config in g.configs]
    # NOTE(j_luo) Set the `gpus` argument to 0 since `dask-cuda` would set `CUDA-VISIBLE_DEVICES` automatically to the actual gpu id.
    cmdl = [cmd + ' --gpus 0' for cmd in cmdl]
    cluster = LocalCUDACluster(n_workers=len(g.gpus),
                               threads_per_worker=1,
                               CUDA_VISIBLE_DEVICES=g.gpus)
    client = Client(cluster)
    client.register_worker_plugin(WorkerSetup())
    jobs = list()
    for cmd in cmdl:
        jobs.append(client.submit(main_distr, cmd))
    progress(*jobs)

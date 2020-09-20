import sys
from multiprocessing import current_process

from dask.distributed import Client, Worker, WorkerPlugin, progress
from dask_cuda import LocalCUDACluster

from dev_misc import add_argument, g, parse_args, show_args
from dev_misc.arglib import disable_duplicate_check, test_with_arguments
from sound_law.main import main, setup


def main_distr(config: str):
    sys.argv = [sys.argv[0]] + f'--config {config} --gpus 0'.split()
    main()


class WorkerSetup(WorkerPlugin):

    def setup(self, worker: Worker):
        setup()


if __name__ == '__main__':
    add_argument('configs', nargs='+', dtype=str, msg='All configs to run.')
    add_argument('gpus', nargs='+', dtype=int, msg='GPUs to use.')
    parse_args()
    cluster = LocalCUDACluster(n_workers=len(g.gpus),
                               threads_per_worker=1,
                               CUDA_VISIBLE_DEVICES=g.gpus)
    client = Client(cluster)
    client.register_worker_plugin(WorkerSetup())
    jobs = list()
    for config in g.configs:
        jobs.append(client.submit(main_distr, config))
    progress(*jobs)

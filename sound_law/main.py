from dev_misc import initiate, parse_args, show_args
from dev_misc.devlib.named_tensor import patch_named_tensors
from sound_law.train.manager import OnePairManager


def run():
    manager = OnePairManager()
    manager.run()


if __name__ == '__main__':
    initiate(logger=True, log_level=True, gpus=True, random_seed=True, commit_id=True, log_dir=True)
    patch_named_tensors()
    parse_args()
    show_args()
    run()

from dev_misc import add_argument, g, initiate, parse_args, show_args
from dev_misc.devlib.named_tensor import patch_named_tensors
from sound_law.config import reg
from sound_law.train.manager import OnePairManager, OneToManyManager

add_argument('task', dtype=str, default='one_pair', choices=['one_pair', 'one_to_many'], msg='Which task to execute.')


def run():
    if g.task == 'one_pair':
        manager = OnePairManager()
    else:
        manager = OneToManyManager()
    manager.run()


if __name__ == '__main__':
    initiate(reg, logger=True, log_level=True, gpus=True, random_seed=True, commit_id=True, log_dir=True)
    patch_named_tensors()
    parse_args()
    show_args()
    run()

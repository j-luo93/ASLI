from dev_misc.trainlib.base_trainer import BaseTrainer


class MonoTrainer(BaseTrainer):


    def __init__(self, model, tasks, task_weights, main_tname, stage_tnames=None, evaluator=None, check_tname='check', check_interval=None, eval_tname='eval', eval_interval=None, save_tname='save', save_interval=None):

        super().__init__(model, tasks, task_weights, main_tname, stage_tnames=stage_tnames, evaluator=evaluator, check_tname=check_tname,
                         check_interval=check_interval, eval_tname=eval_tname, eval_interval=eval_interval, save_tname=save_tname, save_interval=save_interval)

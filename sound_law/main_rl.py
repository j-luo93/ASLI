import sys

from torch.optim import Adam

from dev_misc.trainlib import init_params
from sound_law.main import setup
from sound_law.model.base_model import get_emb_params
from sound_law.model.module import CharEmbedding, EmbParams, PhonoEmbedding
from sound_law.rl.action import SoundChangeAction, SoundChangeActionSpace
from sound_law.rl.data_loader import EntireBatchOnePairDataLoader
from sound_law.rl.env import SoundChangeEnv, TrajectoryCollector
from sound_law.rl.model import ActorCritic
from sound_law.rl.trainer import ActorCriticTrainer
from sound_law.rl.trajectory import VocabState
from sound_law.train.manager import OneToManyManager

if __name__ == "__main__":
    initiator = setup()
    sys.argv = 'main.py --config ZSLatSpaPhono --test_keep_ratio 0.0002'.split()
    initiator.run()
    man = OneToManyManager()
    setting = man.dl_reg.get_setting_by_name('test@spa')
    dl = EntireBatchOnePairDataLoader(setting, man.cog_reg, man.lang2id)
    batch = dl.get_next_batch()
    init_state = VocabState.from_seqs(batch.src_seqs)
    end_state = VocabState.from_seqs(batch.tgt_seqs)
    # FIXME(j_luo)  This should be reused. It is also used in trainer.py
    # HACK(j_luo)
    init_state.ids.rename_(batch='word')
    end_state.ids.rename_(batch='word')

    action_space = SoundChangeActionSpace(man.tgt_abc)
    emb_params = get_emb_params(len(man.tgt_abc), man.tgt_abc.pfm, man.tgt_abc.special_ids)
    emb = PhonoEmbedding.from_params(emb_params)
    model = ActorCritic(emb, action_space, end_state)
    init_params(model, 'xavier_uniform')
    collector = TrajectoryCollector(512, max_rollout_length=50, truncate_last=True)
    env = SoundChangeEnv(end_state)
    trainer = ActorCriticTrainer(model, [setting], [1.0], 'STEP',
                                 collector=collector,
                                 env=env)
    trainer.set_optimizer(Adam, lr=1e-3)
    trainer.train_one_step(dl)

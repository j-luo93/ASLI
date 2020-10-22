import sys

from torch.optim import Adam

from dev_misc.trainlib import init_params
from sound_law.main import setup
from sound_law.model.base_model import get_emb_params
from sound_law.model.module import CharEmbedding, EmbParams, PhonoEmbedding
from sound_law.train.manager import OnePairManager
from sound_law.train.trainer import PolicyGradientTrainer

if __name__ == "__main__":
    initiator = setup()
    sys.argv = 'main.py --config OPEngFake'.split()
    initiator.run()
    man = OnePairManager()
    setting = man.dl_reg.get_setting_by_name('test@spa')
    dl = EntireBatchOnePairDataLoader(setting, man.cog_reg, man.lang2id)
    trainer.set_optimizer(Adam, lr=1e-3)
    trainer.train_one_step(dl)

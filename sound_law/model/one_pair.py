from .base_model import BaseModel


class OnePairModel(BaseModel):

    def _prepare_lang_emb(self, batch):
        return None

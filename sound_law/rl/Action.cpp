#include <Action.h>

Action::Action(long action_id, long before_id, long after_id)
{
    this->action_id = action_id;
    this->before_id = before_id;
    this->after_id = after_id;
}

IdSeq Action::apply_to(const IdSeq &vocab_i)
{
    IdSeq ret = IdSeq(vocab_i.size());
    replace_copy(vocab_i.begin(), vocab_i.end(), ret.begin(), this->before_id, this->after_id);
    return ret;
}

#include <Action.h>

Action::Action(long action_id, long before_id, long after_id)
{
    this->action_id = action_id;
    this->before_id = before_id;
    this->after_id = after_id;
}

pair<bool, IdSeq> Action::apply_to(IdSeq vocab_i)
{
    IdSeq ret = IdSeq(vocab_i.size());
    replace_copy(vocab_i.begin(), vocab_i.end(), ret.begin(), this->before_id, this->after_id);
    bool replaced = find(vocab_i.begin(), vocab_i.end(), this->before_id) != vocab_i.end();
    return pair<bool, IdSeq>(replaced, ret);
}

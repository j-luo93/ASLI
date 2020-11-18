#include <Action.h>

Action::Action(long action_id, long before_id, long after_id)
{
    this->action_id = action_id;
    this->before_id = before_id;
    this->after_id = after_id;
}

Action::Action(long action_id, long before_id, long after_id, long pre_id)
{
    this->action_id = action_id;
    this->before_id = before_id;
    this->after_id = after_id;
    this->pre_id = pre_id;
}

IdSeq Action::apply_to(const IdSeq &id_seq)
{
    if (this->pre_id == -1)
        return this->apply_to_uncond(id_seq);
    else
        return this->apply_to_pre(id_seq);
}

IdSeq Action::apply_to_uncond(const IdSeq &id_seq)
{
    IdSeq ret = IdSeq(id_seq.size());
    replace_copy(id_seq.begin(), id_seq.end(), ret.begin(), this->before_id, this->after_id);
    return ret;
}

IdSeq Action::apply_to_pre(const IdSeq &id_seq)
{
    IdSeq ret = IdSeq();
    for (long i = 1; i < id_seq.size(); ++i)
        if ((id_seq[i] == this->before_id) and (id_seq[i - 1] == this->pre_id))
            ret.push_back(this->after_id);
        else
            ret.push_back(id_seq[i]);
    return ret;
}

#include <Action.h>

Action::Action(action_t action_id, abc_t before_id, abc_t after_id)
{
    this->action_id = action_id;
    this->before_id = before_id;
    this->after_id = after_id;
}

Action::Action(action_t action_id, abc_t before_id, abc_t after_id, abc_t pre_id)
{
    this->action_id = action_id;
    this->before_id = before_id;
    this->after_id = after_id;
    this->pre_id = pre_id;
}

IdSeq Action::apply_to(const IdSeq &id_seq)
{
    if (this->is_conditional())
        return this->apply_to_pre(id_seq);
    else
        return this->apply_to_uncond(id_seq);
}

IdSeq Action::apply_to_uncond(const IdSeq &id_seq)
{
    IdSeq ret = IdSeq(id_seq.size());
    replace_copy(id_seq.begin(), id_seq.end(), ret.begin(), this->before_id, this->after_id);
    return ret;
}

IdSeq Action::apply_to_pre(const IdSeq &id_seq)
{
    IdSeq ret = IdSeq{id_seq[0]};
    for (size_t i = 1; i < id_seq.size(); ++i)
        if ((id_seq[i] == this->before_id) and (id_seq[i - 1] == this->pre_id))
            ret.push_back(this->after_id);
        else
            ret.push_back(id_seq[i]);
    return ret;
}

bool Action::is_conditional()
{
    return this->pre_id != NULL_abc;
}
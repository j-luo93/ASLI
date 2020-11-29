#include <Action.h>

Action::Action(action_t action_id,
               abc_t before_id,
               abc_t after_id,
               vector<abc_t> pre_cond = vector<abc_t>(),
               vector<abc_t> post_cond = vector<abc_t>()) : action_id(action_id),
                                                            before_id(before_id),
                                                            after_id(after_id),
                                                            pre_cond(pre_cond),
                                                            post_cond(post_cond)
{
    this->num_pre = pre_cond.size();
    this->num_post = post_cond.size();
    assert(this->num_pre <= 2);
    assert(this->num_post <= 2);
}

IdSeq Action::apply_to(const IdSeq &id_seq)
{
    IdSeq ret = IdSeq();
    for (size_t i = 0; i < this->num_pre; ++i)
        ret.push_back(id_seq.at(i));
    for (size_t i = this->num_pre; i < id_seq.size() - this->num_post; ++i)
    {
        bool applied = (id_seq.at(i) == this->before_id);
        if (applied)
            for (size_t j = 0; j < this->num_pre; ++j)
                if (id_seq.at(i - this->num_pre + j) != this->pre_cond.at(j))
                {
                    applied = false;
                    break;
                }
        if (applied)
            for (size_t j = 0; j < this->num_post; ++j)
                if (id_seq.at(i + j + 1) != this->post_cond.at(j))
                {
                    applied = false;
                    break;
                }

        if (applied)
            ret.push_back(this->after_id);
        else
            ret.push_back(id_seq.at(i));
    }
    for (size_t i = 0; i < this->num_post; ++i)
        ret.push_back(id_seq.at(id_seq.size() - this->num_post + i));
    return ret;
}

abc_t Action::get_pre_id()
{
    return (this->num_pre == 0) ? NULL_abc : this->pre_cond.back();
}

abc_t Action::get_d_pre_id()
{
    return (this->num_pre < 2) ? NULL_abc : this->pre_cond.front();
}

abc_t Action::get_post_id()
{
    return (this->num_post == 0) ? NULL_abc : this->post_cond.front();
}

abc_t Action::get_d_post_id()
{
    return (this->num_post < 2) ? NULL_abc : this->post_cond.back();
}

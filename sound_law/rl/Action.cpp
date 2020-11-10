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

ActionSpace::ActionSpace()
{
    this->actions = vector<Action *>();
}

void ActionSpace::register_action(long before_id, long after_id)
{
    // FIXME(j_luo) there is no duplicate check here. Should be done by python?
    long action_id = this->actions.size();
    Action *action = new Action(action_id, before_id, after_id);
    this->actions.push_back(action);
}

Action *ActionSpace::get_action(long action_id)
{
    return this->actions[action_id];
}

vector<bool> ActionSpace::get_action_mask(TreeNode *node)
{
    if (not node->action_mask.empty())
        return node->action_mask;
    vector<bool> ret = vector<bool>(this->size(), false);
    for (long i = 0; i < node->size(); ++i)
    {
        IdSeq id_seq = node->vocab_i[i];
        // FIXME(j_luo) cache?
        for (long j = 0; j < this->size(); ++j)
        {
            Action *action = this->actions[j];
            bool replaced = action->apply_to(id_seq).first;
            if (replaced)
                ret[j] = true;
        }
    }
    node->action_mask = ret;
    return ret;
}

long ActionSpace::size()
{
    return this->actions.size();
}
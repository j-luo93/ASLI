#include <ActionSpace.h>

ActionSpace::ActionSpace()
{
    this->actions = vector<Action *>();
}

void ActionSpace::register_action(long before_id, long after_id)
{
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
    for (long i = 0; i < this->size(); ++i)
    {
        Action *action = this->actions[i];
        for (long j = 0; j < node->size(); ++j)
        {
            IdSeq id_seq = node->vocab_i[j];
            bool replaced = action->apply_to(id_seq).first;
            if (replaced)
            {
                ret[i] = true;
                break;
            }
        }
    }
    node->action_mask = ret;
    return ret;
}

long ActionSpace::size()
{
    return this->actions.size();
}
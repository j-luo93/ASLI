#include "Action.hpp"

void ActionSpace::register_edges(abc_t before_id, abc_t after_id)
{
    edges[before_id].push_back(after_id);
}

action_t ActionSpace::get_action_id(const Action &action)
{
    if (a2i.find(action) != a2i.end())
        return a2i.at(action);

    action_t action_id = (action_t)a2i.size();
    a2i[action] = action_id;
    return action_id;
}
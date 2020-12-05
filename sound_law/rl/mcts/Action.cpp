#include "Action.hpp"

void ActionSpace::register_edges(abc_t before_id, abc_t after_id)
{
    edges[before_id].push_back(after_id);
}
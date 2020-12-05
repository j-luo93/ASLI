#pragma once

#include "common.hpp"

using Action = std::array<abc_t, 6>;
using action_t = uint32_t; // for actions

class ActionSpace
{
public:
    void register_edges(abc_t, abc_t);
    action_t get_action_id(const Action &);

private:
    boost::unordered_map<abc_t, std::vector<abc_t>> edges;
    boost::unordered_map<Action, action_t> a2i; // mapping from actions to action ids;
};
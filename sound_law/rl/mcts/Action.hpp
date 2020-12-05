#pragma once

#include "common.hpp"

using Action = std::array<abc_t, 6>;

class ActionSpace
{
public:
    void register_edges(abc_t, abc_t);

private:
    boost::unordered_map<abc_t, std::vector<abc_t>> edges;
};
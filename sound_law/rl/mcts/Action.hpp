#pragma once

#include "common.hpp"
#include "SiteGraph.hpp"

class TreeNode;

class ActionSpace
{
public:
    ActionSpace(SiteSpace *);

    SiteSpace *site_space;

    void register_edges(abc_t, abc_t);
    action_t get_action_id(const Action &);
    void set_action_allowed(TreeNode *);

private:
    boost::unordered_map<abc_t, std::vector<abc_t>> edges;
    boost::unordered_map<Action, action_t> a2i; // mapping from actions to action ids;
};
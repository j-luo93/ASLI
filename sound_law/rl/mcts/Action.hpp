#pragma once

#include "common.hpp"
#include "SiteGraph.hpp"
#include "Word.hpp"
#include "ctpl.h"

class TreeNode;

class ActionSpace
{
public:
    ActionSpace(SiteSpace *, WordSpace *, int);

    SiteSpace *site_space;
    WordSpace *word_space;

    void register_edge(abc_t, abc_t);
    void set_action_allowed(TreeNode *);
    void set_action_allowed(const std::vector<TreeNode *> &);

private:
    void set_action_allowed_no_lock(TreeNode *);

    ctpl::thread_pool tp;
    UMap<abc_t, std::vector<abc_t>> edges;
    boost::shared_mutex actions_mtx;
};
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
    action_t get_action_id(const Action &);
    Action get_action(action_t);
    void set_action_allowed(TreeNode *);
    void set_action_allowed(const std::vector<TreeNode *> &);
    size_t size();
    std::vector<abc_t> expand_a2i();

private:
    action_t safe_get_action_id(const Action &);
    void set_action_allowed_no_lock(TreeNode *);

    ctpl::thread_pool tp;
    UMap<abc_t, std::vector<abc_t>> edges;
    ActionMap<action_t> a2i; // mapping from actions to action ids;
    std::vector<Action> actions;
    boost::shared_mutex actions_mtx;
    std::vector<abc_t> a2i_cache;
};
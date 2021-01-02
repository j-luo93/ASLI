#pragma once

#include "common.hpp"
#include "SiteGraph.hpp"
#include "Word.hpp"
#include "ctpl.h"

class TreeNode;

class ActionSpace
{
public:
    ActionSpace(SiteSpace *, WordSpace *, float, int, int);

    SiteSpace *site_space;
    WordSpace *word_space;
    const float dist_threshold;
    const int site_threshold;
    const int num_threads;

    void register_edge(abc_t, abc_t);
    void set_action_allowed(TreeNode *);
    void set_action_allowed(const std::vector<TreeNode *> &);

private:
    ctpl::thread_pool tp;
    UMap<abc_t, std::vector<abc_t>> edges;

    void find_potential_actions(TreeNode *,
                                std::vector<uai_t> &,
                                std::vector<std::vector<int>> &,
                                std::vector<uai_t> &,
                                std::vector<std::vector<int>> &);
    void apply_new_word_no_lock(Word *,
                                const std::vector<int> &,
                                const std::vector<std::vector<uai_t>> &);
    void set_action_allowed_no_lock(TreeNode *,
                                    const std::vector<uai_t> &,
                                    const std::vector<std::vector<int>> &);
};
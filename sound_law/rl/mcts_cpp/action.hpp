#pragma once

#include "common.hpp"
#include "site.hpp"
#include "word.hpp"
#include "tree_node.hpp"

using ParaAction = vec<uai_t>;
using ParaOrder = vec<vec<int>>;
using UnseenAction = vec<vec<uai_t>>;
using GraphOutput = tup<ParaAction, ParaOrder, UnseenAction>;

class ActionSpace
{
    UMap<abc_t, vec<abc_t>> edges;

    bool match(abc_t, abc_t);

public:
    SiteSpace *site_space;
    WordSpace *word_space;
    const float dist_threshold;
    const int site_threshold;
    Timer &timer = Timer::getInstance();

    ActionSpace(SiteSpace *, WordSpace *, float, int);

    void register_edge(abc_t, abc_t);
    void set_action_allowed(Pool *, const vec<TreeNode *> &);
    void set_action_allowed(TreeNode *);
    void apply_action(Word *&, Word *, uai_t);
    IdSeq apply_action(const IdSeq &, uai_t);
    vec<uai_t> get_similar_actions(uai_t);
};
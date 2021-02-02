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
    UMap<abc_t, vec<pair<SpecialType, abc_t>>> edges;
    vec<bool> vowel_mask;
    vec<abc_t> vowel_base;
    vec<Stress> vowel_stress;
    vec<abc_t> stressed_vowel;
    vec<abc_t> unstressed_vowel;
    abc_t glide_j;
    abc_t glide_w;

    bool match(abc_t, abc_t);
    void apply_actions(vec<Word *> &, Word *, usi_t, const vec<uai_t> &);

public:
    SiteSpace *site_space;
    WordSpace *word_space;
    const float dist_threshold;
    const int site_threshold;

    ActionSpace(SiteSpace *, WordSpace *, float, int);

    void register_edge(abc_t, abc_t);
    void register_cl_map(abc_t, abc_t); // compensatory length edge.
    void set_vowel_info(const vec<bool> &, const vec<abc_t> &, const vec<Stress> &, const vec<abc_t> &, const vec<abc_t> &);
    void set_glide_info(abc_t, abc_t);
    void set_action_allowed(Pool *, const vec<TreeNode *> &);
    void set_action_allowed(TreeNode *);
    // void apply_action(Word *&, Word *, uai_t);
    IdSeq apply_action(const IdSeq &, uai_t);
    vec<uai_t> get_similar_actions(uai_t);
    int locate_edge_index(abc_t, abc_t);
};
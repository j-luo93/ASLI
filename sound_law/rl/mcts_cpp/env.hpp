#pragma once

#include "common.hpp"
#include "node.hpp"
#include "word.hpp"
#include "action.hpp"

struct EnvOpt
{
    VocabIdSeq start_ids;
    VocabIdSeq end_ids;
    float final_reward;
    float step_penalty;
};

class Mcts;

class Env
{
    friend class Mcts;

    ActionSpace *action_space;
    WordSpace *word_space;

    TreeNode *apply_action(TreeNode *, const Subpath &);

public:
    Env(const EnvOpt &, const ActionSpaceOpt &, const WordSpaceOpt &);

    const EnvOpt opt;
    TreeNode *start;
    TreeNode *end;

    inline void register_permissible_change(abc_t before, abc_t after) { action_space->register_permissible_change(before, after); };
    inline void evaluate(TreeNode *node, const vec<vec<float>> &meta_priors, const vec<float> &special_priors) { action_space->evaluate(node, meta_priors, special_priors); };
    inline float get_edit_dist(const IdSeq &seq1, const IdSeq &seq2) { return word_space->get_edit_dist(seq1, seq2); };
    inline TreeNode *apply_action(TreeNode *node,
                                  abc_t before,
                                  abc_t after,
                                  abc_t pre,
                                  abc_t d_pre,
                                  abc_t post,
                                  abc_t d_post,
                                  SpecialType special_type) { return action_space->apply_action(node, before, after, pre, d_pre, post, d_post, special_type); };
    inline void register_cl_map(abc_t before, abc_t after) { action_space->register_cl_map(before, after); };
    inline void register_gbj_map(abc_t before, abc_t after) { action_space->register_gbj_map(before, after); };
    inline void register_gbw_map(abc_t before, abc_t after) { action_space->register_gbw_map(before, after); };
    inline void clear_stats(TreeNode *node, bool recursive) { action_space->clear_stats(node, recursive); };
};
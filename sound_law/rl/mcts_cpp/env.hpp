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
    inline void evaluate(TreeNode *node, const MetaPriors &meta_priors, const vec<float> &special_priors) { action_space->evaluate(node, meta_priors, special_priors); };
    inline float get_edit_dist(const IdSeq &seq1, const IdSeq &seq2) { return word_space->get_edit_dist(seq1, seq2); };
    inline TreeNode *apply_action(TreeNode *node,
                                  abc_t before_id,
                                  abc_t after_id,
                                  abc_t pre_id,
                                  abc_t d_pre_id,
                                  abc_t post_id,
                                  abc_t d_post_id,
                                  SpecialType special_type) { return action_space->apply_action(node, before_id, after_id, pre_id, d_pre_id, post_id, d_post_id, special_type); };
};
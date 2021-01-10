#pragma once

#include "common.hpp"
#include "action.hpp"
#include "tree_node.hpp"

class Env
{
public:
    ActionSpace *action_space;
    WordSpace *word_space;
    TreeNode *start;
    TreeNode *end;
    const float final_reward;
    const float step_penalty;

public:
    Env(ActionSpace *, WordSpace *, const VocabIdSeq &, const VocabIdSeq &, float, float);

    TreeNode *apply_action(TreeNode *, int, uai_t);
};
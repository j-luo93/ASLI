#pragma once

#include "common.hpp"
#include "Action.hpp"
#include "Word.hpp"

class TreeNode;

class Env
{
public:
    Env(WordSpace *, ActionSpace *, const VocabIdSeq &, float, float);

    TreeNode *apply_action(TreeNode *, action_t, action_t);

    WordSpace *word_space;
    ActionSpace *action_space;
    TreeNode *start;
    TreeNode *end;
    const float final_reward;
    const float step_penalty;

private:
    TreeNode *apply_action(TreeNode *, const Action &, action_t, action_t);
};
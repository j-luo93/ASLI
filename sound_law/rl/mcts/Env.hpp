#pragma once

#include "common.hpp"
#include "Action.hpp"
#include "Word.hpp"

class TreeNode;

class Env
{
public:
    Env(WordSpace *, ActionSpace *, const VocabIdSeq &, const VocabIdSeq &);

    TreeNode *apply_action(TreeNode *, const Action &);

    WordSpace *word_space;
    ActionSpace *action_space;
    TreeNode *start;
    TreeNode *end;
};
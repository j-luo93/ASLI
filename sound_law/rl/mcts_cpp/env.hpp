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

    WordSpace *word_space;

    TreeNode *apply_action(TreeNode *, const Subpath &);

public:
    Env(const EnvOpt &, const ActionSpaceOpt &, const WordSpaceOpt &);

    const EnvOpt opt;
    TreeNode *start;
    TreeNode *end;
    ActionSpace *action_space;
};
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

class Env
{
    WordSpace *word_space;

public:
    Env(const EnvOpt &env_opt, const WordSpaceOpt &ws_opt);

    EnvOpt opt;
    TreeNode *start;
    TreeNode *end;
    ActionSpace *action_space;

    TreeNode *apply_action(TreeNode *, const Subpath &);
};
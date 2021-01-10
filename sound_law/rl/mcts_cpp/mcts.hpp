#pragma once

#include "common.hpp"
#include "env.hpp"

class Mcts
{
    Pool *tp;

public:
    Env *env;
    const float puct_c;
    const int game_count;
    const float virtual_loss;
    const int num_threads;

    Mcts(Env *, float, int, float, int);

    vec<TreeNode *> select(TreeNode *, int, int);
    void backup(const vec<TreeNode *> &, const vec<float> &);
    uai_t play(TreeNode *);
    void set_logging_options(int, bool);
};
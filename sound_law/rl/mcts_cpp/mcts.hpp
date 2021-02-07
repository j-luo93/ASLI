#pragma once

#include "common.hpp"
#include "env.hpp"
#include "node.hpp"

struct MctsOpt
{
    float puct_c;
    int game_count;
    float virtual_loss;
    int num_threads;
};

class Mcts
{
    Pool *tp;
    Env *env;

    TreeNode *select_single_thread(TreeNode *, int);

public:
    MctsOpt opt;

    Mcts(Env *, const MctsOpt &);

    vec<TreeNode *> select(TreeNode *, int, int);
    void backup(const vec<TreeNode *> &, const vec<float> &);
};

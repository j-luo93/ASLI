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
    float heur_c;
};

struct Edge
{
    BaseNode *s0;
    ChosenChar a;
    BaseNode *s1;
};

struct Path
{
    list<Subpath> subpaths;
    vec<TreeNode *> tree_nodes;

    // Return all edges (s0, a, s1) from the descendant to the root.
    vec<Edge> get_edges_to_root() const;
};

class Mcts
{
    Pool *tp;
    Env *env;

    Path select_single_thread(TreeNode *, int) const;

public:
    MctsOpt opt;

    Mcts(Env *, const MctsOpt &);

    vec<Path> select(TreeNode *, int, int) const;
    void backup(const vec<Path> &, const vec<float> &) const;
    TreeNode *play(TreeNode *node) { return node->play(); };
};

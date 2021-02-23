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

class Path
{
private:
    vec<Subpath> subpaths;
    vec<TreeNode *> tree_nodes;
    int depth;

public:
    // FIXME(j_luo) This is hacky for cython.
    Path() = default;
    Path(TreeNode *, const int);

    // Return all edges (s0, a, s1) from the descendant to the root.
    vec<Edge> get_edges_to_root() const;
    int get_depth() const;
    // Append both subpath and tree node at the back.
    void append(const Subpath &, TreeNode *);
    // Whether a new node adds a circle.
    bool forms_a_circle(TreeNode *) const;
};

class Mcts
{
    Pool *tp;
    Env *env;

    Path select_single_thread(TreeNode *, const int, const int) const;

public:
    MctsOpt opt;

    Mcts(Env *, const MctsOpt &);

    vec<Path> select(TreeNode *, const int, const int, const int) const;
    void backup(const vec<Path> &, const vec<float> &) const;
    TreeNode *play(TreeNode *node) { return node->play(); };
};

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
    bool add_noise;
    bool use_num_misaligned;
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
    Path(const Path &);
    Path(TreeNode *, const int);

    // Return all edges (s0, a, s1) from the descendant to the root.
    vec<Edge> get_edges_to_root() const;
    int get_depth() const;
    // Append both subpath and tree node at the back.
    void append(const Subpath &, TreeNode *);
    // Whether a new node adds a circle.
    bool forms_a_circle(TreeNode *) const;

    vec<BaseNode *> get_all_nodes() const;
    vec<size_t> get_all_chosen_indices() const;
    vec<abc_t> get_all_chosen_actions() const;
    void merge(const Path &);
    TreeNode *get_last_node() const;
};

class Mcts
{
    Pool *tp;
    Env *env;

    Path select_single_thread(TreeNode *, const int, const int, const Path &) const;

public:
    MctsOpt opt;

    Mcts(Env *, const MctsOpt &);

    vec<Path> select(TreeNode *, const int, const int, const int) const;
    vec<Path> select(TreeNode *, const int, const int, const int, const Path &) const;
    void backup(const vec<Path> &, const vec<float> &) const;
    inline Path play(TreeNode *node, int start_depth, PlayStrategy ps)
    {
        auto ret = Path(node, start_depth);
        auto play_ret = node->play(ps);
        ret.append(play_ret.second, play_ret.first);
        for (const auto node : ret.get_all_nodes())
            env->cache.put_persistent(node);
        return ret;
    };
};

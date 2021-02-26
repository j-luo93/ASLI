#pragma once

#include "word.hpp"
#include "node.hpp"

struct ActionSpaceOpt
{
    abc_t null_id;
    abc_t emp_id;
    abc_t sot_id;
    abc_t eot_id;
    abc_t any_id;
    abc_t any_s_id;
    abc_t any_uns_id;
    abc_t glide_j;
    abc_t glide_w;
    int site_threshold;
    float dist_threshold;
};

class Env;
class Mcts;

class ActionSpace
{
    friend Env;
    friend Mcts;

    WordSpace *word_space;

    Subpath get_best_subpath(TreeNode *, float, int, float, float, bool);
    MiniNode *get_mini_node(TreeNode *, BaseNode *, const ChosenChar &, ActionPhase, bool);
    IdSeq change_id_seq(const IdSeq &, const vec<size_t> &, abc_t, SpecialType);
    void update_affected(BaseNode *, abc_t, int, size_t, map<abc_t, size_t> &, bool);
    // void update_affected(BaseNode *, const IdSeq &, int, size_t, int, map<abc_t, size_t> &);

    // Methods for expanding nodes.
    void expand(TreeNode *);
    void expand(MiniNode *, const Subpath &, bool, bool);
    void expand_before(MiniNode *, BaseNode *, int);
    void expand_special_type(MiniNode *, BaseNode *, int, abc_t, bool);
    void expand_after(MiniNode *, BaseNode *, int, bool, bool, bool);
    void expand_pre(MiniNode *, BaseNode *, int, bool, bool);
    void expand_d_pre(MiniNode *, BaseNode *, int, bool, bool, bool);
    void expand_post(MiniNode *, BaseNode *, int, bool, bool);
    void expand_normal(MiniNode *, BaseNode *, int, int, bool, bool, bool);
    void expand_null(MiniNode *, BaseNode *, int);
    bool expand_null_only(MiniNode *, BaseNode *, int);

    void evaluate(MiniNode *);
    // This will create a new tree node without checking first if the child exists. Use `apply_action` in `Env` if checking is needed.
    TreeNode *apply_new_action(TreeNode *, const Subpath &);
    TreeNode *apply_action(TreeNode *, abc_t, abc_t, abc_t, abc_t, abc_t, abc_t, SpecialType);
    void register_permissible_change(abc_t, abc_t);
    void register_cl_map(abc_t, abc_t);
    void register_gbj_map(abc_t, abc_t);
    void register_gbw_map(abc_t, abc_t);
    void evaluate(TreeNode *, const vec<vec<float>> &, const vec<float> &);

    ActionSpace(WordSpace *, const ActionSpaceOpt &);
    map<abc_t, vec<abc_t>> permissible_changes;
    map<abc_t, abc_t> cl_map;
    map<abc_t, abc_t> gbj_map;
    map<abc_t, abc_t> gbw_map;

    void expand_stats(BaseNode *);
    void clear_stats(BaseNode *, bool);
    void clear_priors(BaseNode *, bool);
    void prune(BaseNode *, bool);
    void add_noise(TreeNode *, const vec<vec<float>> &, const vec<float> &, float);

public:
    const ActionSpaceOpt opt;
};
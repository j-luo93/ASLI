#pragma once

#include "common.hpp"
#include "word.hpp"

class Env;
class Mcts;

class TreeNode
{
    friend class Env;
    friend class Mcts;

    std::mutex score_mtx;
    std::mutex neighbor_mtx;
    bool played = false;

    TreeNode(const vec<Word *> &, int);
    TreeNode(const vec<Word *> &, const pair<int, uai_t> &, TreeNode *, bool);

    void common_init();
    int select(float, int, float);
    void backup(float, int, float);

public:
    vec<Word *> words;
    pair<int, uai_t> prev_action;
    TreeNode *parent_node = nullptr;
    bool stopped = false;
    bool done;
    int depth;
    float dist;
    vec<uai_t> action_allowed;
    vec<float> prior;
    vec<visit_t> action_count;
    vec<float> total_value;
    visit_t visit_count = 0;
    ActionMap<TreeNode *> neighbors;
    ActionMap<float> rewards;
    float max_value;
    int max_index;
    uai_t max_action_id;

    bool is_leaf();
    vec<float> get_scores(float);
    int get_best_i(float);
    void expand(const vec<float> &);
    std::string str();
    IdSeq get_id_seq(int);
    size_t size();
    size_t get_num_descendants();
    void clear_stats(bool = false);
    void add_noise(const vec<float> &, float);
};

//This class only holds static data without links to other nodes.
class DetachedTreeNode
{
public:
    DetachedTreeNode(TreeNode *);

    vec<uai_t> action_allowed;
    VocabIdSeq vocab_i;

    IdSeq get_id_seq(int);
    size_t size();
};
#pragma once

#include "common.hpp"

class Env;
class ActionSpace;
class Word;

class TreeNode
{
public:
    static boost::mutex cls_mtx;
    static tn_cnt_t cls_cnt;

    friend class Env;
    friend class ActionSpace;

    void debug(bool = true);
    bool is_leaf();
    std::vector<float> get_scores(float);
    int get_best_i(float);
    int select(float, int, float);
    void expand(const std::vector<float> &);
    void backup(float, float, int, float);
    void play();
    std::list<std::pair<uai_t, float>> get_path();
    void add_noise(const std::vector<float> &, float);
    IdSeq get_id_seq(int);
    void clear_stats(bool = false);
    size_t size();
    size_t get_num_descendants(); // Including `this`.
    size_t clear_cache(float);    // Returns num of descendants.

    // Basic members.
    tn_cnt_t idx;
    std::vector<Word *> words;
    bool stopped = false;
    bool done;
    float dist;
    std::vector<uai_t> action_allowed;
    TreeNode *parent_node = nullptr;
    std::pair<int, uai_t> prev_action;
    ActionMap<TreeNode *> neighbors;
    ActionMap<float> rewards;
    // Stored after evaluation.
    std::vector<float> prior;
    // Game stats.
    std::vector<visit_t> action_count;
    std::vector<float> total_value;
    visit_t visit_count;
    float max_value;
    int max_index;
    uai_t max_action_id;
    bool played;

private:
    void common_init();
    TreeNode(const std::vector<Word *> &);
    TreeNode(const std::vector<Word *> &, const std::pair<int, uai_t> &, TreeNode *, bool);
    void virtual_backup(int, int, float);
    boost::mutex exclusive_mtx;
    boost::shared_mutex neighbor_mtx;
};

//This class only holds static data without links to other nodes.
class DetachedTreeNode
{
public:
    DetachedTreeNode(TreeNode *);

    IdSeq get_id_seq(int);
    size_t size();

    vec<uai_t> action_allowed;
    VocabIdSeq vocab_i;
};
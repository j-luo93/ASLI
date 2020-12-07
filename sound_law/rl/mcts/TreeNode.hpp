#pragma once

#include "Word.hpp"

class Env;

class TreeNode
{
public:
    friend class Env;

    void debug();
    bool is_leaf();
    std::vector<float> get_scores(float);
    action_t get_best_i(float);
    action_t select(float, int, float);
    void expand(const std::vector<float> &);
    void backup(float, float, int, float);
    void play();
    std::list<std::pair<action_t, float>> get_path();
    void add_noise(const std::vector<float> &, float);
    IdSeq get_id_seq(int);

    // Basic members.
    std::vector<Word *> words;
    bool done;
    float dist;
    std::vector<action_t> action_allowed;
    TreeNode *parent_node = nullptr;
    std::pair<action_t, action_t> prev_action;
    boost::unordered_map<action_t, TreeNode *> neighbors;
    boost::unordered_map<action_t, float> rewards;
    // Stored after evaluation.
    std::vector<float> prior;
    // Game stats.
    std::vector<visit_t> action_count;
    std::vector<float> total_value;
    visit_t visit_count;
    float max_value;
    int max_index;
    action_t max_action_id;
    bool played;
    void clear_stats();
    size_t size();

private:
    void common_init();
    TreeNode(const std::vector<Word *> &);
    TreeNode(const std::vector<Word *> &, const std::pair<action_t, action_t> &, TreeNode *);
    void virtual_backup(action_t, int, float);
    boost::mutex select_mtx;
    boost::shared_mutex neighbor_mtx;
};

//This class only holds static data without links to other nodes.
class DetachedTreeNode
{
public:
    IdSeq get_id_seq(int);
    size_t size();

    std::vector<action_t> action_allowed;
    VocabIdSeq vocab_i;
};
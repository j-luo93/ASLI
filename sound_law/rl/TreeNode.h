#pragma once

#include <common.h>

using namespace std;

class TreeNode
{
public:
    static node_t instance_cnt;
    static mutex cls_mtx;
    static vector<vector<cost_t>> dist_mat;
    static cost_t ins_cost;
    static bool max_mode;

    static void set_dist_mat(vector<vector<cost_t>> &);
    static void set_max_mode(bool);

    TreeNode(const VocabIdSeq &);
    TreeNode(const VocabIdSeq &, TreeNode *);
    TreeNode(const VocabIdSeq &, TreeNode *, action_t, action_t, TreeNode *);

    void add_edge(action_t, const pair<TreeNode *, float> &);
    bool has_acted(action_t);
    size_t size();
    void lock();
    void unlock();
    bool is_leaf();
    action_t get_best_i(float);
    void expand(const vector<float> &);
    void virtual_backup(action_t, int, float);
    void backup(float, float, int, float);
    void reset();
    void play();
    list<pair<action_t, float>> get_path();
    vector<float> get_scores(float);
    void clear_subtree();
    void add_noise(const vector<float> &, float);
    size_t get_num_allowed();

    VocabIdSeq vocab_i;
    TreeNode *end_node;
    TreeNode *parent_node;
    pair<action_t, action_t> prev_action;
    dist_t dist_to_end;
    unordered_map<action_t, pair<TreeNode *, float>> edges;
    vector<action_t> action_allowed;
    vector<float> prior;
    vector<visit_t> action_count;
    visit_t visit_count;
    vector<float> total_value;
    vector<float> max_value;
    bool done;
    bool played;
    node_t idx;

private:
    mutex mtx;
    unique_lock<mutex> ulock;
};

using Edge = pair<TreeNode *, float>;

class DetachedTreeNode
{
public:
    // This only holds the data, without the pointers to parents and children, or visit counts.
    DetachedTreeNode(TreeNode *);

    VocabIdSeq vocab_i;
    pair<action_t, action_t> prev_action;
    dist_t dist_to_end;
    vector<action_t> action_allowed;
    bool done;

    size_t size();
};
#pragma once

#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <common.h>

using namespace std;

class TreeNode
{
public:
    static long instance_cnt;
    static mutex cls_mtx;

    TreeNode(VocabIdSeq);
    TreeNode(VocabIdSeq, TreeNode *, vector<long>);
    TreeNode(VocabIdSeq, TreeNode *, long, long, TreeNode *, vector<long>);

    void add_edge(long, pair<TreeNode *, float>);
    bool has_acted(long);
    long size();
    void lock();
    void unlock();
    bool is_leaf();
    long get_best_i(float);
    void expand(vector<float>);
    void virtual_backup(long, long, float);
    void backup(float, float, long, float);
    void reset();
    void play();
    list<pair<long, float>> get_path();
    vector<float> get_scores(float);
    void clear_subtree();
    void add_noise(vector<float>, float);
    long get_num_allowed();

    VocabIdSeq vocab_i;
    TreeNode *end_node;
    TreeNode *parent_node;
    pair<long, long> prev_action;
    long dist_to_end;
    unordered_map<long, pair<TreeNode *, float>> edges;
    vector<long> action_allowed;
    vector<float> prior;
    vector<long> action_count;
    long visit_count;
    vector<float> total_value;
    bool done;
    bool played;
    long idx;

private:
    mutex mtx;
    unique_lock<mutex> ulock;
};

using Edge = pair<TreeNode *, float>;
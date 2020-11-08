#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>

using namespace std;

// FIXME(j_luo) Probably need list for insertion speed.
using IdSeq = vector<long>;
using VocabIdSeq = vector<IdSeq>;

class TreeNode
{
public:
    TreeNode(VocabIdSeq);
    TreeNode(VocabIdSeq, TreeNode *);
    TreeNode(VocabIdSeq, TreeNode *, long, TreeNode *);

    void add_edge(long, TreeNode *);
    bool has_acted(long);
    long size();
    void lock();
    void unlock();
    bool is_leaf();
    long get_best_action_id(float);
    void expand(vector<float>);
    void virtual_backup(long, long, float);
    void backup(float, long, float);

    VocabIdSeq vocab_i;
    TreeNode *end_node;
    TreeNode *parent_node;
    long prev_action;
    long dist_to_end;
    unordered_map<long, TreeNode *> edges;
    vector<float> prior;
    vector<long> action_count;
    long visit_count;
    vector<float> total_value;

private:
    mutex mtx;
    unique_lock<mutex> ulock;
};
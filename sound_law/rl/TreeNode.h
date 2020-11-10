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

    void add_edge(long, pair<TreeNode *, float>);
    bool has_acted(long);
    long size();
    void lock();
    void unlock();
    bool is_leaf();
    long get_best_action_id(float);
    void expand(vector<float>, vector<bool>);
    void virtual_backup(long, long, float);
    void backup(float, float, long, float);
    void reset();
    void play();

    VocabIdSeq vocab_i;
    TreeNode *end_node;
    TreeNode *parent_node;
    long prev_action;
    long dist_to_end;
    unordered_map<long, pair<TreeNode *, float>> edges;
    vector<bool> action_mask;
    vector<float> prior;
    vector<long> action_count;
    long visit_count;
    vector<float> total_value;
    bool done;
    bool played;

private:
    mutex mtx;
    unique_lock<mutex> ulock;
};

using Edge = pair<TreeNode *, float>;
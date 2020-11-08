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

    void add_edge(long, TreeNode *);
    bool has_acted(long);
    long size();
    void lock();
    void unlock();
    bool is_leaf();

    VocabIdSeq vocab_i;
    TreeNode *end_node;
    long dist_to_end;
    unordered_map<long, TreeNode *> edges;
    vector<float> prior;

private:
    mutex mtx;
    unique_lock<mutex> ulock;
};
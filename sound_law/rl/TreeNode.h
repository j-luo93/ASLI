#pragma once

#include <vector>
#include <unordered_map>

using namespace std;

using uint = unsigned int;
using IdSeq = vector<uint>;
using VocabIdSeq = vector<IdSeq>;


class TreeNode
{
public:
    TreeNode(VocabIdSeq, TreeNode *);

    void add_edge(uint, TreeNode *);
    bool has_acted(uint);
    uint size();

    VocabIdSeq vocab_i;
    TreeNode *end_node;
    unsigned long dist_to_end;
    unordered_map<uint, TreeNode *> edges;
};
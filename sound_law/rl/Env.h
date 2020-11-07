#pragma once

#include <stdlib.h>
#include <TreeNode.h>
#include <Action.h>
#include <assert.h>
#include <iostream>

uint edit_distance(IdSeq seq1, IdSeq seq2)
{
    uint l1 = seq1.size();
    uint l2 = seq2.size();
    uint **dist = (uint **)malloc((l1 + 1) * sizeof(uint));
    for (uint i = 0; i < l1 + 1; ++i)
        dist[i] = (uint *)malloc((l2 + 1));

    for (uint i = 0; i < l1 + 1; ++i)
        dist[i][0] = i;
    for (uint i = 0; i < l2 + 1; ++i)
        dist[0][i] = i;

    for (uint i = 1; i < l1 + 1; ++i)
        for (uint j = 1; j < l2 + 1; ++j)
        {
            uint sub_cost = seq1[i] == seq2[j] ? 0 : 1;
            dist[i][j] = min(dist[i - 1][j - 1] + sub_cost, min(dist[i - 1][j], dist[i][j - 1]) + 1);
        }
    return dist[l1][l2];
};

unsigned long node_distance(TreeNode *node1, TreeNode *node2)
{
    uint l1 = node1->size();
    uint l2 = node2->size();
    assert(l1 == l2);
    unsigned long ret = 0;
    for (uint i = 0; i < l1; ++i)
    {
        uint dist = edit_distance(node1->vocab_i[i], node2->vocab_i[i]);
        ret = ret + dist;
    }
    return ret;
};

class Env
{
public:
    Env(TreeNode *, TreeNode *);

    TreeNode *step(TreeNode *, Action *);

    TreeNode *init_node;
    TreeNode *end_node;
};
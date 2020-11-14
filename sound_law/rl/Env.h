#pragma once

#include <stdlib.h>
#include <TreeNode.h>
#include <Action.h>
#include <ActionSpace.h>
#include <assert.h>
#include <iostream>

long edit_distance(IdSeq seq1, IdSeq seq2)
{
    long l1 = seq1.size();
    long l2 = seq2.size();
    long **dist = (long **)malloc((l1 + 1) * sizeof(long **));
    for (long i = 0; i < l1 + 1; ++i)
        dist[i] = (long *)malloc((l2 + 1) * sizeof(long *));

    for (long i = 0; i < l1 + 1; ++i)
        dist[i][0] = i;
    for (long i = 0; i < l2 + 1; ++i)
        dist[0][i] = i;

    for (long i = 1; i < l1 + 1; ++i)
        for (long j = 1; j < l2 + 1; ++j)
        {
            long sub_cost = seq1[i - 1] == seq2[j - 1] ? 0 : 1;
            dist[i][j] = min(dist[i - 1][j - 1] + sub_cost, min(dist[i - 1][j], dist[i][j - 1]) + 1);
        }
    long ret = dist[l1][l2];
    for (long i = 0; i < l1 + 1; ++i)
        free(dist[i]);
    free(dist);
    return ret;
};

long node_distance(TreeNode *node1, TreeNode *node2)
{
    long l1 = node1->size();
    long l2 = node2->size();
    assert(l1 == l2);
    unsigned long ret = 0;
    for (long i = 0; i < l1; ++i)
    {
        long dist = edit_distance(node1->vocab_i[i], node2->vocab_i[i]);
        ret = ret + dist;
    }
    return ret;
};

class Env
{
public:
    Env(TreeNode *, TreeNode *, ActionSpace *, float, float);

    Edge step(TreeNode *, Action *);

    TreeNode *init_node;
    TreeNode *end_node;
    ActionSpace *action_space;

private:
    float final_reward;
    float step_penalty;
    float starting_dist;
};
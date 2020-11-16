#pragma once

#include <stdlib.h>
#include <TreeNode.h>
#include <Action.h>
#include <ActionSpace.h>
#include <assert.h>
#include <iostream>

long node_distance(TreeNode *node1, TreeNode *node2, const vector<vector<long>> &dist_mat, long ins_cost)
{
    long l1 = node1->size();
    long l2 = node2->size();
    assert(l1 == l2);
    unsigned long ret = 0;
    for (long i = 0; i < l1; ++i)
    {
        long dist = edit_distance(node1->vocab_i[i], node2->vocab_i[i], dist_mat, ins_cost);
        ret = ret + dist;
    }
    return ret;
};

class Env
{
public:
    Env(TreeNode *, TreeNode *, ActionSpace *, float, float);

    Edge step(TreeNode *, long, Action *);

    TreeNode *init_node;
    TreeNode *end_node;
    ActionSpace *action_space;

private:
    float final_reward;
    float step_penalty;
    float starting_dist;
};
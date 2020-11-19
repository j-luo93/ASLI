#pragma once

#include <stdlib.h>
#include <TreeNode.h>
#include <Action.h>
#include <ActionSpace.h>
#include <assert.h>

dist_t node_distance(TreeNode *node1, TreeNode *node2, const vector<vector<cost_t>> &dist_mat, cost_t ins_cost)
{
    size_t l1 = node1->size();
    size_t l2 = node2->size();
    assert(l1 == l2);
    dist_t ret = 0;
    for (size_t i = 0; i < l1; ++i)
    {
        ret += edit_distance(node1->vocab_i[i], node2->vocab_i[i], dist_mat, ins_cost);
    }
    return ret;
};

class Env
{
public:
    Env(TreeNode *, TreeNode *, ActionSpace *, float, float);

    Edge step(TreeNode *, action_t, Action *);

    TreeNode *init_node;
    TreeNode *end_node;
    ActionSpace *action_space;

private:
    float final_reward;
    float step_penalty;
    float starting_dist;
};
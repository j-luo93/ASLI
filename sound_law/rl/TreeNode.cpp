#include <TreeNode.h>
#include <Env.h>
#include <iostream>

TreeNode::TreeNode(VocabIdSeq vocab_i, TreeNode *end_node = nullptr)
{
    this->vocab_i = vocab_i;
    this->end_node = end_node;
    this->dist_to_end = (end_node == nullptr) ? 0 : node_distance(this, end_node);
};

void TreeNode::add_edge(uint action_id, TreeNode *child)
{
    // This will replace the old edge if it exists. Always call `has_acted` first.
    this->edges[action_id] = child;
}

bool TreeNode::has_acted(uint action_id)
{
    return this->edges.find(action_id) != this->edges.end();
}

uint TreeNode::size()
{
    return (uint)this->vocab_i.size();
}

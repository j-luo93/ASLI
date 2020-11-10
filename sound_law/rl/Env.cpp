#include <Env.h>
#include <algorithm>

Env::Env(TreeNode *init_node, TreeNode *end_node)
{
    this->init_node = init_node;
    this->end_node = end_node;
};

TreeNode *Env::step(TreeNode *node, Action *action)
{
    assert(node->end_node != nullptr);
    if (node->has_acted(action->action_id))
    {
        return node->edges[action->action_id];
    }

    VocabIdSeq vocab_i = VocabIdSeq(node->size());
    for (long i = 0; i < node->size(); ++i)
    {
        vocab_i[i] = action->apply_to(node->vocab_i[i]).second;
    };
    TreeNode *new_node = new TreeNode(vocab_i, node->end_node, action->action_id, node);
    node->add_edge(action->action_id, new_node);
    return new_node;
};

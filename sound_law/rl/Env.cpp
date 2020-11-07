#include <Env.h>
#include <algorithm>

Env::Env(TreeNode *init_node, TreeNode *end_node)
{
    this->init_node = init_node;
    this->end_node = end_node;
};

TreeNode *Env::step(TreeNode *node, Action *action)
{
    VocabIdSeq vocab_i = VocabIdSeq(node->size());
    for (uint i = 0; i < node->size(); ++i)
    {
        vocab_i[i] = IdSeq(node->vocab_i[i].size());
        replace_copy(node->vocab_i[i].begin(), node->vocab_i[i].end(), vocab_i[i].begin(), action->before_id, action->after_id);
    };
    return new TreeNode(vocab_i, node->end_node);
};

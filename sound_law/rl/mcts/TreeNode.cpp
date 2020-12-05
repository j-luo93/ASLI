#include "TreeNode.hpp"

TreeNode::TreeNode(const std::vector<Word *> &words) : words(words) {}

void TreeNode::debug()
{

    std::cerr << "Debug TreeNode:\n";
    for (Word *word : words)
    {
        std::cerr << "---------\n";
        word->debug();
    }
}

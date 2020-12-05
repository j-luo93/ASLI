#pragma once

#include "Word.hpp"

class TreeNode
{
public:
    TreeNode(const std::vector<Word *> &);

    std::vector<Word *> words;
};
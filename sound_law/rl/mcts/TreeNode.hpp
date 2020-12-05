#pragma once

#include "Word.hpp"

class Env;

class TreeNode
{
public:
    friend class Env;

    void debug();

    std::vector<Word *> words;
    boost::unordered_map<action_t, TreeNode *> neighbors;

private:
    TreeNode(const std::vector<Word *> &);
};

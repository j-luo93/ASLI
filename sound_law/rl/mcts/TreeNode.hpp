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
    std::vector<action_t> action_allowed;

private:
    TreeNode(const std::vector<Word *> &);
};

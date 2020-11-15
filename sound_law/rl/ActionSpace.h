#pragma once

#include <Action.h>
#include <TreeNode.h>
#include <Word.h>

class ActionSpace
{
public:
    ActionSpace();

    void register_action(long, long);
    Action *get_action(long);
    vector<long> get_action_allowed(const VocabIdSeq &);
    long size();

private:
    vector<Action *> actions;
    unordered_map<string, Word *> word_cache;
    unordered_map<long, vector<long>> uni_map;
    mutex mtx;
};
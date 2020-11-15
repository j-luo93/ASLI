#pragma once

#include <Action.h>
#include <TreeNode.h>

class ActionSpace
{
public:
    ActionSpace();

    void register_action(long, long);
    Action *get_action(long);
    vector<long> get_action_allowed(VocabIdSeq);
    long size();

private:
    vector<Action *> actions;
};
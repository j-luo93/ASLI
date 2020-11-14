#pragma once

#include <Action.h>
#include <TreeNode.h>

class ActionSpace
{
public:
    ActionSpace();

    void register_action(long, long);
    Action *get_action(long);
    vector<bool> get_action_mask(VocabIdSeq);
    long size();

private:
    vector<Action *> actions;
};
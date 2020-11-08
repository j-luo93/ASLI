#pragma once

#include <TreeNode.h>
#include <algorithm>

class Action
{
public:
    Action(long, long, long);

    pair<bool, IdSeq> apply_to(IdSeq);

    long action_id;
    long before_id;
    long after_id;
    // FIXME(j_luo) some cache is needed here.
};

class ActionSpace
{
public:
    ActionSpace();

    void register_action(long, long);
    Action *get_action(long);
    vector<bool> get_action_mask(TreeNode *);
    long size();

private:
    vector<Action *> actions;
};
#pragma once

#include <TreeNode.h>

class Action
{
public:
    Action(long, long, long);

    long action_id;
    long before_id;
    long after_id;
};
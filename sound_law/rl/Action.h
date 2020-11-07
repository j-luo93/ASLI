#pragma once

#include <TreeNode.h>

class Action
{
public:
    Action(uint, uint, uint);

    uint action_id;
    uint before_id;
    uint after_id;
};
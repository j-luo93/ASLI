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

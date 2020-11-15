#pragma once

#include <TreeNode.h>
#include <algorithm>

class Action
{
public:
    Action(long, long, long);

    IdSeq apply_to(const IdSeq &);

    long action_id;
    long before_id;
    long after_id;
    // TODO(j_luo) some cache is needed here.
};

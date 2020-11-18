#pragma once

#include <TreeNode.h>
#include <algorithm>

class Action
{
public:
    Action(long, long, long);
    Action(long, long, long, long);

    IdSeq apply_to(const IdSeq &);

    long action_id;
    long before_id;
    long after_id;
    long pre_id = -1;

private:
    IdSeq apply_to_uncond(const IdSeq &);
    IdSeq apply_to_pre(const IdSeq &);
};

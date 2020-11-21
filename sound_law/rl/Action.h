#pragma once

#include <TreeNode.h>
#include <algorithm>

class Action
{
public:
    Action(action_t, abc_t, abc_t);
    Action(action_t, abc_t, abc_t, abc_t);

    IdSeq apply_to(const IdSeq &);
    bool is_conditional();

    action_t action_id;
    abc_t before_id;
    abc_t after_id;
    abc_t pre_id = NULL_abc;

private:
    IdSeq apply_to_uncond(const IdSeq &);
    IdSeq apply_to_pre(const IdSeq &);
};

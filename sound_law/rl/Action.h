#pragma once

#include <TreeNode.h>
#include <algorithm>

class Action
{
public:
    Action(abc_t, abc_t, abc_t);
    Action(abc_t, abc_t, abc_t, abc_t);

    IdSeq apply_to(const IdSeq &);
    bool is_conditional();

    abc_t action_id;
    abc_t before_id;
    abc_t after_id;
    abc_t pre_id = NULL_abc;

private:
    IdSeq apply_to_uncond(const IdSeq &);
    IdSeq apply_to_pre(const IdSeq &);
};

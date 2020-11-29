#pragma once

#include <TreeNode.h>
#include <algorithm>

class Action
{
public:
    Action(action_t, abc_t, abc_t, vector<abc_t>, vector<abc_t>);

    IdSeq apply_to(const IdSeq &);

    action_t action_id;
    abc_t before_id;
    abc_t after_id;
    vector<abc_t> pre_cond;
    vector<abc_t> post_cond;
    size_t num_pre;
    size_t num_post;

    abc_t get_pre_id();
    abc_t get_post_id();
    abc_t get_d_pre_id();
    abc_t get_d_post_id();
};

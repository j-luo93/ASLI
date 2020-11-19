#pragma once

#include <Action.h>
#include <TreeNode.h>
#include <Word.h>

class ActionSpace
{
public:
    static bool use_conditional;

    static void set_conditional(bool);

    ActionSpace();

    void register_action(abc_t, abc_t);
    void register_action(abc_t, abc_t, abc_t);
    Action *get_action(action_t);
    vector<action_t> get_action_allowed(const VocabIdSeq &);
    size_t size();
    void clear_cache();
    size_t get_cache_size();

private:
    vector<Action *> actions;
    unordered_map<string, Word *> word_cache;
    unordered_map<abc_t, vector<action_t>> uni_map;
    unordered_map<abc_t, unordered_map<abc_t, vector<action_t>>> pre_map;
    mutex mtx;
};
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

    void register_action(long, long);
    void register_action(long, long, long);
    Action *get_action(long);
    vector<long> get_action_allowed(const VocabIdSeq &);
    long size();
    void clear_cache();
    long get_cache_size();

private:
    vector<Action *> actions;
    unordered_map<string, Word *> word_cache;
    unordered_map<long, vector<long>> uni_map;
    unordered_map<long, unordered_map<long, vector<long>>> pre_map;
    mutex mtx;
};
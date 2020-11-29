#pragma once

#include <Action.h>
#include <TreeNode.h>
#include <Word.h>
#include <Site.h>

class ActionSpace
{
public:
    static bool use_conditional;

    static void set_conditional(bool);

    ActionSpace();

    vector<Action *> actions;

    void register_action(abc_t, abc_t, const vector<abc_t> &, const vector<abc_t> &);
    Action *get_action(action_t);
    vector<action_t> get_action_allowed(const VocabIdSeq &);
    size_t size();
    void clear_cache();
    size_t get_cache_size();

private:
    unordered_map<WordKey, Word *> word_cache;
    unordered_map<SiteKey, vector<action_t>> site_map;
    mutex mtx;
};
#pragma once

#include <common.h>
#include <Site.h>

class Word
{
public:
    Word(const IdSeq &, const WordKey &);

    WordKey key;
    vector<Site> sites;
    // unordered_set<action_t> action_allowed_uncond;
    // unordered_set<abc_t> uni_keys;
    // unordered_map<abc_t, unordered_set<abc_t>> pre_keys;
};

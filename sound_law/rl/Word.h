#pragma once

#include <common.h>

class Word
{
public:
    Word(const IdSeq &);

    string key;
    unordered_set<action_t> action_allowed_uncond;
    unordered_set<abc_t> uni_keys;
    unordered_map<abc_t, unordered_set<abc_t>> pre_keys;
};

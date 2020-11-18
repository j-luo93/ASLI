#pragma once

#include <common.h>

class Word
{
public:
    Word(const IdSeq &);

    string key;
    unordered_set<long> action_allowed_uncond;
    unordered_set<long> uni_keys;
    unordered_map<long, unordered_set<long>> pre_keys;
};

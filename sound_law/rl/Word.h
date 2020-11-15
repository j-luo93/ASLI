#pragma once

#include <common.h>

class Word
{
public:
    Word(const IdSeq &);

    string key;
    unordered_set<long> action_allowed;
    unordered_set<long> uni_keys;
};

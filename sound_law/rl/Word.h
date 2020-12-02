#pragma once

#include <common.h>
#include <Site.h>

class Word
{
public:
    Word(const IdSeq &, const WordKey &);

    WordKey key;
    vector<SiteNode *> site_roots;
};

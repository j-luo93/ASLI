#pragma once

#include <common.h>

class SiteNode;

class Word
{
public:
    static VocabIdSeq end_words;
    static void set_end_words(const VocabIdSeq &);

    Word(const IdSeq &, const WordKey &, size_t);

    vector<abc_t> id_seq;
    WordKey key;
    vector<SiteNode *> site_roots;
    size_t order;
};

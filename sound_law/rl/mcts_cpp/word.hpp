#pragma once

#include "common.hpp"

class Word
{
    friend class WordSpace;

    Word(const IdSeq &);

    // FIXME(j_luo) optimize this later.

    paramap<int, float> dists;

public:
    const IdSeq id_seq;

    float get_edit_dist(int);
};

struct WordSpaceOpt
{
    vec<vec<float>> dist_mat;
    float ins_cost;
};

class WordSpace
{
    friend class ActionSpace;
    friend class Env;

    paramap<IdSeq, Word *> words;
    vec<Word *> end_words;

public:
    const WordSpaceOpt opt;

    WordSpace(const VocabIdSeq &, const WordSpaceOpt &);

    Word *get_word(const IdSeq &);
    void set_edit_dist(Word *, int);
};
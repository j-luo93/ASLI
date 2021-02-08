#pragma once

#include "common.hpp"

class TreeNode;

class Word
{
    friend class WordSpace;
    friend class TreeNode;

    Word(const IdSeq &, const IdSeq &, const vec<size_t> &);

    // FIXME(j_luo) optimize this later.
    paramap<int, float> dists;

    float get_edit_dist_at(int);

public:
    const IdSeq id_seq;
    const IdSeq vowel_seq;
    const vec<size_t> id2vowel;
};

struct WordSpaceOpt
{
    vec<vec<float>> dist_mat;
    float ins_cost;
    vec<bool> is_vowel;
    vec<Stress> unit_stress;
    vec<abc_t> unit2base;
    vec<abc_t> unit2stressed;
    vec<abc_t> unit2unstressed;
};

class WordSpace
{
    friend class ActionSpace;
    friend class Env;

    WordSpace(const VocabIdSeq &, const WordSpaceOpt &);

    paramap<IdSeq, Word *> words;
    vec<Word *> end_words;

    void set_edit_dist_at(Word *, int);
    Word *get_word(const IdSeq &);
    float get_edit_dist(const IdSeq &, const IdSeq &);

public:
    const WordSpaceOpt opt;
};
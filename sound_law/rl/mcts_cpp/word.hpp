#pragma once

#include "common.hpp"

class TreeNode;

class Word
{
    friend class WordSpace;

    Word(const IdSeq &, const IdSeq &, const vec<size_t> &);

    paramap<int, float> dists;

public:
    const IdSeq id_seq;
    const IdSeq vowel_seq;
    const vec<size_t> id2vowel;

    // Get edit distance at a given `order`.
    float get_edit_dist_at(int) const;
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

    paramap<IdSeq, Word *> words;

public:
    WordSpace(const WordSpaceOpt &, const VocabIdSeq &);

    const WordSpaceOpt opt;
    const vec<Word *> end_words;

    void set_edit_dist_at(Word *, int) const;
    Word *get_word(const IdSeq &);
    vec<Word *> get_words(const VocabIdSeq &);
    float get_edit_dist(const IdSeq &, const IdSeq &) const;
    size_t size() const;
};
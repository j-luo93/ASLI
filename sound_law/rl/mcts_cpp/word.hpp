#pragma once

#include "common.hpp"

class TreeNode;

namespace alignment
{
    const int INSERTED = -1;
};

struct Alignment
{
    vec<size_t> pos_seq1;
    vec<size_t> pos_seq2;
    vec<int> aligned_pos;
};

class Word
{
    friend class WordSpace;

    Word(const IdSeq &, const IdSeq &, const vec<size_t> &);

    paramap<int, float> dists;
    paramap<int, Alignment> almts;

public:
    const IdSeq id_seq;
    const IdSeq vowel_seq;
    const vec<size_t> id2vowel;

    // Get edit distance at a given `order`.
    float get_edit_dist_at(int) const;
    // Get alignment at a given `order`.
    const Alignment &get_almt_at(int) const;
};

struct WordSpaceOpt
{
    vec<vec<float>> dist_mat;
    float ins_cost;
    bool use_alignment;
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
    float get_edit_dist(const IdSeq &, const IdSeq &, Alignment &) const;
    size_t size() const;
    // Get misalignment score for `word` with the end state at `order` at `position`.
    float get_misalignment_score(const Word *, int, size_t, abc_t) const;
};
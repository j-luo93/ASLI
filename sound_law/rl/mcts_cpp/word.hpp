#pragma once

#include "common.hpp"
#include "site.hpp"

class Word
{
    friend class WordSpace;

    Word(const IdSeq &, const vec<SiteNode *> &, size_t);

public:
    const IdSeq id_seq;
    ParaMap<usi_t, Word *> neighbors;
    // ActionMap<Word *> neighbors;
    vec<SiteNode *> site_roots;
    std::string str();
    DistTable dists;
};

class ActionSpace;
class Env;

class WordSpace
{
    friend class ActionSpace;
    friend class Env;

    // IdSeqMap<Word *> words;
    ParaMap<IdSeq, Word *> words;
    float get_edit_dist(Word *, Word *);
    vec<Word *> end_words;

public:
    SiteSpace *site_space;
    const vec<vec<float>> dist_mat;
    const float ins_cost;
    Timer &timer = Timer::getInstance();

    WordSpace(SiteSpace *, const vec<vec<float>> &, float);

    void get_words(Pool *, vec<Word *> &, const vec<IdSeq> &, bool = false, size_t = 0);
    void get_word(Word *&, const IdSeq &, size_t = 0);
    size_t size() const;
    void set_end_words(const vec<Word *> &);
    float safe_get_dist(Word *, int);
    float get_edit_dist(const IdSeq &, const IdSeq &);
};
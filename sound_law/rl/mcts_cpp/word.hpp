#pragma once

#include "common.hpp"
#include "site.hpp"

class Word
{
    friend class WordSpace;

    Word(const IdSeq &, const IdSeq &, const vec<SiteNode *> &, const vec<SiteNode *> &, size_t);

public:
    const IdSeq id_seq;
    const IdSeq vowel_seq;
    ParaMap<usi_t, vec<Word *>> neighbors;
    ParaMap<usi_t, vec<Word *>> vowel_neighbors;

    vec<SiteNode *> site_roots;
    vec<SiteNode *> vowel_site_roots;
    std::string str();
    DistTable dists;
};

class ActionSpace;
class Env;

class WordSpace
{
    friend class ActionSpace;
    friend class Env;

    ParaMap<IdSeq, Word *> words;
    vec<Word *> end_words;

    float get_edit_dist(Word *, Word *);
    vec<SiteNode *> get_site_roots(const IdSeq &);

public:
    SiteSpace *site_space;
    const vec<vec<float>> dist_mat;
    const float ins_cost;

    WordSpace(SiteSpace *, const vec<vec<float>> &, float);

    void get_words(Pool *, vec<Word *> &, const vec<IdSeq> &, bool = false, size_t = 0);
    void get_word(Word *&, const IdSeq &, size_t = 0);
    size_t size() const;
    void set_end_words(const vec<Word *> &);
    float safe_get_dist(Word *, int);
    float get_edit_dist(const IdSeq &, const IdSeq &);
};
#pragma once

#include "common.hpp"
#include "Site.hpp"

class SiteNode;

class Word
{
public:
    friend class WordSpace;

    size_t size();
    void debug();

    const IdSeq id_seq;
    float dist;
    boost::unordered_map<Action, Word *> neighbors;
    std::vector<SiteNode *> site_roots;

private:
    Word(const IdSeq &, const std::vector<SiteNode *> &, float);
};

class WordSpace
{
public:
    WordSpace(SiteSpace *, const std::vector<std::vector<float>> &, float, const VocabIdSeq &);

    SiteSpace *site_space;
    std::vector<std::vector<float>> dist_mat;
    float ins_cost;
    std::vector<Word *> end_words;

    Word *get_word(const IdSeq &, int, bool);
    // Apply action to the word. Order information is needed to compute the edit distance (against end word).
    Word *apply_action(Word *, const Action &, int);
    size_t size();

private:
    boost::unordered_map<IdSeq, Word *> words;
    float get_edit_dist(const IdSeq &, const IdSeq &);
};

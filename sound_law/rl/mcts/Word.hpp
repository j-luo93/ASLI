#pragma once

#include "common.hpp"
#include "Site.hpp"

class SiteNode;
class TreeNode;

class Word
{
public:
    friend class WordSpace;
    friend class TreeNode;

    size_t size();
    void debug();

    const IdSeq id_seq;
    bool done;
    ActionMap<Word *> neighbors;
    std::vector<SiteNode *> site_roots;

private:
    Word(const IdSeq &, const std::vector<SiteNode *> &, int, float, bool);
    boost::shared_mutex neighbor_mtx;
    UMap<int, float> dists;
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
    Word *apply_action(Word *, uai_t, int);
    Word *apply_action_no_lock(Word *, uai_t, int);
    size_t size();
    float get_dist(Word *, int);

private:
    UMap<IdSeq, Word *> words;
    float get_edit_dist(const IdSeq &, const IdSeq &);
    boost::shared_mutex words_mtx;
    bool match(abc_t, abc_t);
};

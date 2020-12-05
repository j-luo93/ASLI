#pragma once

#include "common.hpp"
#include "Action.hpp"
#include "Site.hpp"

class SiteNode;

class Word
{
public:
    friend class WordSpace;

    size_t size();
    void debug();

    const IdSeq id_seq;
    boost::unordered_map<Action, Word *> neighbors;
    std::vector<SiteNode *> site_roots;

private:
    Word(const IdSeq &, const std::vector<SiteNode *> &);
};

class WordSpace
{
public:
    WordSpace(SiteSpace *);

    SiteSpace *site_space;

    Word *get_word(const IdSeq &);
    Word *apply_action(Word *, const Action &);
    size_t size();

private:
    boost::unordered_map<IdSeq, Word *> words;
};

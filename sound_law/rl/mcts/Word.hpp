#pragma once

#include "common.hpp"
#include "Action.hpp"

class Word
{
public:
    IdSeq id_seq;
    boost::unordered_map<Action, Word *> neighbors;
    size_t size();
    void debug();

    friend class WordSpace;

private:
    Word(const IdSeq &);
};

class WordSpace
{
public:
    Word *get_word(const IdSeq &);
    Word *apply_action(Word *, const Action &);
    size_t size();

private:
    boost::unordered_map<IdSeq, Word *> words;
};

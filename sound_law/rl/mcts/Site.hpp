#pragma once

#include "common.hpp"

using Site = std::array<abc_t, 5>;

class Word;

class SiteNode
{
public:
    SiteNode(const Site &);

    void debug();

    const Site site;
    SiteNode *lchild = nullptr;
    SiteNode *rchild = nullptr;
    friend class SiteSpace;

private:
    bool visited = false;

private:
    SiteNode();
};

class SiteSpace
{
public:
    // // Link word to the site (and its descendants)
    // void link_word(Word *, const Site &);
    // Given a root site, return its associated node (and generate the subgraph that leads from it along the way).
    SiteNode *get_node(abc_t, abc_t, abc_t, abc_t, abc_t);

private:
    boost::unordered_map<Site, SiteNode *> nodes;
    // Given a root node, return the list of SiteNodes (including the root).
    std::vector<SiteNode *> get_descendants(SiteNode *);
};

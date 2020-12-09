#pragma once

#include "common.hpp"
#include <mutex>

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
    SiteNode();
};

class SiteSpace
{
public:
    SiteNode *get_node(abc_t, abc_t, abc_t, abc_t, abc_t);

private:
    UMap<Site, SiteNode *> nodes;
    boost::shared_mutex nodes_mtx;
};

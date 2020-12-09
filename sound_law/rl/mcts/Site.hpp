#pragma once

#include "common.hpp"
#include <mutex>

class Word;

class SiteNode
{
public:
    SiteNode(usi_t);

    const usi_t site;
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
    UMap<usi_t, SiteNode *> nodes;
    boost::shared_mutex nodes_mtx;
};

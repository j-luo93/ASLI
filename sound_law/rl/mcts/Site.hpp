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
    SiteNode *lxchild = nullptr;
    SiteNode *rchild = nullptr;
    SiteNode *rxchild = nullptr;
    friend class SiteSpace;

private:
    SiteNode();
};

class SiteSpace
{
public:
    SiteSpace(abc_t, abc_t, abc_t, abc_t);
    SiteNode *get_node(abc_t, abc_t, abc_t, abc_t, abc_t);

    const abc_t sot_id;
    const abc_t eot_id;
    const abc_t any_id;
    const abc_t emp_id;

private:
    UMap<usi_t, SiteNode *> nodes;
    boost::shared_mutex nodes_mtx;
};

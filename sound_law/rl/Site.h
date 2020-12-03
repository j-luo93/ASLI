#pragma once

#include <common.h>

using Site = tuple<abc_t, abc_t, abc_t, abc_t, abc_t>;

void print_site(const Site &site)
{
    cerr << get<0>(site) << ' ' << get<1>(site) << ' ' << get<2>(site) << ' ' << get<3>(site) << ' ' << get<4>(site) << '\n';
}

class Word;

class SiteNode
{
public:
    static mutex cls_mtx;
    static unordered_map<Site, SiteNode *> all_nodes;
    static SiteNode *get_site_node(abc_t, abc_t, abc_t, abc_t, abc_t, Word *);

    void reset();

    Site site;
    SiteNode *lchild = nullptr;
    SiteNode *rchild = nullptr;
    bool visited = false;
    vector<Word *> linked_words;
};

// This is the site nodes with thread-local stats.
class SiteNodeWithStats
{
public:
    SiteNodeWithStats(SiteNode *);

    SiteNode *base;
    size_t num_sites = 0;
    bool visited = false;
    SiteNodeWithStats *lchild = nullptr;
    SiteNodeWithStats *rchild = nullptr;
};

class SiteGraph
{
public:
    SiteGraph();
    ~SiteGraph();

    void *add_node(SiteNode *, SiteNode * = nullptr);
    unordered_map<Site, SiteNodeWithStats *> nodes;

private:
    inline SiteNodeWithStats *get_wrapped_node(SiteNode *);
};

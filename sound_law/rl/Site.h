#pragma once

#include <common.h>

using Site = tuple<abc_t, abc_t, abc_t, abc_t, abc_t>;

class SiteNode
{
public:
    static mutex cls_mtx;
    static unordered_map<Site, SiteNode *> all_nodes;
    static SiteNode *get_site_node(abc_t, abc_t, abc_t, abc_t, abc_t);

    void reset();

    Site site;
    SiteNode *lchild = nullptr;
    SiteNode *rchild = nullptr;
    size_t num_sites = 0;
    size_t in_degree = 0;
    bool visited = false;
};

class SiteGraph
{
public:
    SiteGraph();
    ~SiteGraph();

    void add_node(SiteNode *);
    vector<SiteNode *> get_sources();
    unordered_map<Site, SiteNode *> nodes;

    // private:
    // void add_site(abc_t, const vector<abc_t> &, const vector<abc_t> &, const SiteKey &, SiteNode * = nullptr);
    // void add_site(const Site &, SiteNode * = nullptr);
};

#pragma once

#include <common.h>

using Site = tuple<abc_t, abc_t, abc_t, abc_t, abc_t>;

class SiteNode
{
public:
    SiteNode(const Site &);
    // SiteNode(abc_t, const vector<abc_t> &, const vector<abc_t> &, const SiteKey &);

    Site site;
    vector<SiteNode *> children;
    size_t num_sites = 0;
    size_t in_degree = 0;
    bool visited = false;
};

class SiteGraph
{
public:
    SiteGraph();
    ~SiteGraph();

    void add_site(const Site &, SiteNode * = nullptr);
    vector<SiteNode *> get_sources();
    unordered_map<Site, SiteNode *> nodes;

    // private:
    // void add_site(abc_t, const vector<abc_t> &, const vector<abc_t> &, const SiteKey &, SiteNode * = nullptr);
    // void add_site(const Site &, SiteNode * = nullptr);
};

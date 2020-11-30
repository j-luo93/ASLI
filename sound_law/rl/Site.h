#pragma once

#include <common.h>

class Site
{
public:
    Site(abc_t, const vector<abc_t> &, const vector<abc_t> &, const SiteKey &);
    Site();

    abc_t before_id;
    vector<abc_t> pre_cond;
    vector<abc_t> post_cond;
    SiteKey key;
};

class SiteNode
{
public:
    SiteNode(const Site &);
    SiteNode(abc_t, const vector<abc_t> &, const vector<abc_t> &, const SiteKey &);

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

private:
    unordered_map<SiteKey, SiteNode *> nodes;
    void add_site(abc_t, const vector<abc_t> &, const vector<abc_t> &, const SiteKey &, SiteNode * = nullptr);
};

#pragma once

#include <common.h>

class Site
{
public:
    Site(abc_t, const vector<abc_t> &, const vector<abc_t> &, const SiteKey &);

    abc_t before_id;
    vector<abc_t> pre_cond;
    vector<abc_t> post_cond;
    SiteKey key;
};

class SiteNode
{
public:
    SiteNode(const Site &);

    Site site;
    vector<shared_ptr<SiteNode>> children;
    size_t num_sites = 0;
    size_t in_degree = 0;
    bool visited = false;
};

class SiteGraph
{
public:
    SiteGraph();

    void add_site(const Site &, shared_ptr<SiteNode> = nullptr);
    vector<shared_ptr<SiteNode>> get_sources();

private:
    unordered_map<SiteKey, shared_ptr<SiteNode>> nodes;
};

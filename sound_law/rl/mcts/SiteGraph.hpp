#pragma once

#include "common.hpp"
#include "Site.hpp"

// A wrapper class around SiteNode that includes stats.
class GraphNode
{
public:
    friend class SiteGraph;

    SiteNode *base;
    GraphNode *lchild = nullptr;
    GraphNode *rchild = nullptr;
    int num_sites = 0;

private:
    GraphNode(SiteNode *);
    bool visited = false;
};

class SiteGraph
{
public:
    SiteGraph(SiteSpace *);

    void *add_root(SiteNode *);

    SiteSpace *site_space;
    boost::unordered_map<Site, GraphNode *> nodes;

private:
    GraphNode *generate_subgraph(SiteNode *);
    std::vector<GraphNode *> get_descendants(GraphNode *);
};
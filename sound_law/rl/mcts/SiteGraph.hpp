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
    GraphNode *lxchild = nullptr;
    GraphNode *rxchild = nullptr;
    int num_sites = 0;
    boost::unordered_set<int> linked_words; // the orders for the linked words -- use set since one word might have multiple identical sites

private:
    GraphNode(SiteNode *);
    bool visited = false;
};

class SiteGraph
{
public:
    SiteGraph(SiteSpace *);

    void add_root(SiteNode *, int);

    SiteSpace *site_space;
    UMap<usi_t, GraphNode *> nodes;

private:
    GraphNode *generate_subgraph(SiteNode *);
    std::vector<GraphNode *> get_descendants(GraphNode *);
};
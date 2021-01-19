#pragma once

#include "common.hpp"

class SiteNode
{
    friend class SiteSpace;

    SiteNode(usi_t);

public:
    const usi_t site;
    SiteNode *lchild = nullptr;
    SiteNode *lxchild = nullptr;
    SiteNode *rchild = nullptr;
    SiteNode *rxchild = nullptr;
};

class WordSpace;

class SiteSpace
{
    friend class WordSpace;

    Timer &timer = Timer::getInstance();

    void get_node(SiteNode *&, abc_t, abc_t, abc_t, abc_t, abc_t);

public:
    ParaMap<usi_t, SiteNode *> nodes;
    const abc_t sot_id;
    const abc_t eot_id;
    const abc_t any_id;
    const abc_t emp_id;

    SiteSpace(abc_t, abc_t, abc_t, abc_t);

    size_t size() const;
    void get_node(SiteNode *&, usi_t);
    void get_nodes(Pool *, vec<vec<SiteNode *>> &, const vec<vec<usi_t>> &);
};

// A wrapper class around SiteNode that includes stats.
class GraphNode
{
    friend class SiteGraph;

    bool visited = false;
    GraphNode(SiteNode *);

public:
    SiteNode *base;
    GraphNode *lchild = nullptr;
    GraphNode *rchild = nullptr;
    GraphNode *lxchild = nullptr;
    GraphNode *rxchild = nullptr;
    int num_sites = 0;
    USet<int> linked_words; // the orders for the linked words -- use set since one word might have multiple identical sites
};

class ActionSpace;

class SiteGraph
{
    friend class ActionSpace;

    GraphNode *generate_subgraph(SiteNode *);
    vec<GraphNode *> get_descendants(GraphNode *);
    void visit(vec<GraphNode *> &, GraphNode *);

public:
    UMap<usi_t, GraphNode *> nodes;

    ~SiteGraph();

    GraphNode *add_root(SiteNode *, int);
};
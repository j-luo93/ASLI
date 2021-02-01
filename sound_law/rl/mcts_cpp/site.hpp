#pragma once

#include "common.hpp"

class SiteNode
{
    friend class SiteSpace;

    SiteNode(usi_t);

public:
    const usi_t site;
    vec<SiteNode *> children;
};

class WordSpace;

class SiteSpace
{
    friend class WordSpace;

    vec<bool> vowel_mask;
    vec<int> vowel_base;
    vec<Stress> vowel_stress;

    void get_node(SiteNode *&, abc_t, abc_t, abc_t, abc_t, abc_t);

public:
    ParaMap<usi_t, SiteNode *> nodes;
    const abc_t sot_id;
    const abc_t eot_id;
    const abc_t any_id;
    const abc_t emp_id;
    const abc_t syl_eot_id;
    const abc_t any_s_id;
    const abc_t any_uns_id;

    SiteSpace(abc_t, abc_t, abc_t, abc_t, abc_t, abc_t, abc_t);

    size_t size() const;
    void get_node(SiteNode *&, usi_t);
    void get_nodes(Pool *, vec<vec<SiteNode *>> &, const vec<vec<usi_t>> &);
    void set_vowel_info(const vec<bool> &, const vec<int> &, const vec<Stress> &);
};

// A wrapper class around SiteNode that includes stats.
class GraphNode
{
    friend class SiteGraph;

    bool visited = false;
    GraphNode(SiteNode *);

public:
    SiteNode *base;
    vec<GraphNode *> children;
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
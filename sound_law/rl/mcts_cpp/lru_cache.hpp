#pragma once

#include "common.hpp"
#include "node.hpp"

class CacheNode
{
    friend class LruCache;

    CacheNode(BaseNode *const);

    BaseNode *const base;
};

class LruCache
{
    list<CacheNode> nodes;
    map<BaseNode *, list<CacheNode>::iterator> base2node_it;
    set<BaseNode *> persistent_nodes;
    void evict(BaseNode *const);

public:
    size_t size() const;
    size_t persistent_size() const;
    void evict();
    void put(BaseNode *const);
};
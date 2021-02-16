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
    void evict(const CacheNode &);

public:
    size_t size();
    BaseNode *evict();
    void put(BaseNode *const);
};
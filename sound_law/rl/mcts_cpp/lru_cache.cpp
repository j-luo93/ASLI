#include "lru_cache.hpp"

CacheNode::CacheNode(BaseNode *const base) : base(base) {}

size_t LruCache::size() { return nodes.size(); }

void LruCache::evict(BaseNode *const base)
{
    if (!base2node_it.contains(base))
        return;
    auto node_it = base2node_it[base];
    auto node = *node_it;
    base2node_it.erase(base);
    nodes.erase(node_it);

    for (const auto child : base->children)
        if ((child != nullptr) && (child->get_in_degree() <= 1))
            evict(child);

    EdgeBuilder::disconnect_from_parent(base);
    delete base;
}

BaseNode *LruCache::evict()
{
    auto base = nodes.back().base;
    evict(base);
    return base;
}

void LruCache::put(BaseNode *const base)
{
    assert(base != nullptr);
    if (base2node_it.contains(base))
    {
        auto node_it = base2node_it[base];
        nodes.erase(node_it);
    }
    nodes.push_front(CacheNode(base));
    base2node_it[base] = nodes.begin();
}
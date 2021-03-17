#include "lru_cache.hpp"

CacheNode::CacheNode(BaseNode *base) : base(base) {}

size_t LruCache::size() const { return nodes.size() + persistent_nodes.size(); }

size_t LruCache::persistent_size() const { return persistent_nodes.size(); }

void LruCache::evict(BaseNode *base)
{
    if (!base2node_it.contains(base))
    {
        assert(base2node_it.size() == nodes.size());
        return;
    }
    auto node_it = base2node_it[base];
    auto node = *node_it;
    base2node_it.erase(base);
    nodes.erase(node_it);
    MemoryManager::release(base);
}

void LruCache::evict()
{
    evict(nodes.back().base);
}

void LruCache::put(BaseNode *base)
{
    // Call `put_persisten` if `base` is persistent.
    if (base->is_persistent())
    {
        put_persistent(base);
        return;
    }

    assert(base != nullptr);
    if (base2node_it.contains(base))
    {
        auto node_it = base2node_it[base];
        nodes.erase(node_it);
        SPDLOG_TRACE("LruCache: entry already exists.");
    }
    else
        SPDLOG_TRACE("LruCache: entry doesn't exists");
    nodes.push_front(CacheNode(base));
    base2node_it[base] = nodes.begin();
}

void LruCache::put_persistent(BaseNode *base)
{
    MemoryManager::make_persistent(base);
    // For persistent nodes, we should not evict them ever, therefore not put in the cache.
    if (base2node_it.contains(base))
    {
        auto node_it = base2node_it[base];
        nodes.erase(node_it);
        base2node_it.erase(base);
    }
    persistent_nodes.insert(base);
}
#include "lru_cache.hpp"

CacheNode::CacheNode(BaseNode *const base) : base(base) {}

size_t LruCache::size() { return nodes.size(); }

void LruCache::evict(const CacheNode &cnode)
{
    auto base = cnode.base;
    assert(base2node_it.contains(base));
    auto node_it = base2node_it[base];
    auto node = *node_it;
    base2node_it.erase(base);
    nodes.erase(node_it);

    for (const auto child : base->children)
        if (child != nullptr)
            evict(*base2node_it[child]);
}

BaseNode *LruCache::evict()
{
    // CacheNode *node = nodes.back();
    // BaseNode *base = node->base;
    // base2node_it.erase(base);
    // delete node;
    // nodes.pop_back();

    // for (const auto child : base->children)
    //     if (child != nullptr)
    //         evict(child);
    // if (base->parent != nullptr)
    //     base->parent->children[base->chosen_char.first] = nullptr;
    // delete base;
    auto &cnode = nodes.back();
    evict(cnode);
    return cnode.base;
}

void LruCache::put(BaseNode *const base)
{
    if (base2node_it.contains(base))
    {
        auto node_it = base2node_it[base];
        nodes.erase(node_it);
    }
    nodes.push_front(CacheNode(base));
    base2node_it[base] = nodes.begin();
}
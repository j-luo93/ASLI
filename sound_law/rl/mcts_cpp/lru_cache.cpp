#include "lru_cache.hpp"

CacheNode::CacheNode(BaseNode *const base) : base(base) {}

size_t LruCache::size() { return nodes.size(); }

void LruCache::evict(BaseNode *const base)
{
    if (!base2node_it.contains(base))
    {
        assert(base2node_it.size() == nodes.size());
        std::cerr << "stuck!\n";
        return;
    }
    auto node_it = base2node_it[base];
    auto node = *node_it;
    base2node_it.erase(base);
    nodes.erase(node_it);
    std::cerr << 1 << "\n";

    // for (const auto child : base->children)
    //     if ((child != nullptr) && (child->get_in_degree() <= 1))
    //         evict(child);

    EdgeBuilder::disconnect(base);
    std::cerr << 2 << "\n";
    if (base->is_tree_node() && !base->stopped)
        TreeNode::t_table.remove(static_cast<TreeNode *>(base)->words);
    std::cerr << 3 << "\n";
    // if (base->is_tree_node())
    //     std::cerr << "evicting " << str::from(static_cast<TreeNode *>(base)) << "\n";
    // else
    //     std::cerr << "evicting " << str::from(static_cast<MiniNode *>(base)) << "\n";
    delete base;
}

void LruCache::evict() {
    std::cerr << "evicting...\n" ;
    evict(nodes.back().base);
    std::cerr << "evicted one\n" ;
     }

void LruCache::put(BaseNode *const base)
{
    // For persistent nodes, we should not evict them ever, therefore not put in the cache.
    if (base->persistent)
        return;
    assert(base != nullptr);
    if (base2node_it.contains(base))
    {
        auto node_it = base2node_it[base];
        nodes.erase(node_it);
        SPDLOG_TRACE("LruCache: entry already exists.");
    }
    else
        SPDLOG_TRACE("LruCache: entry doesn't exists");
    // if (base->is_tree_node())
    //     std::cerr << "putting " << str::from(static_cast<TreeNode *>(base)) << "\n";
    // else
    //     std::cerr << "putting " << str::from(static_cast<MiniNode *>(base)) << "\n";
    nodes.push_front(CacheNode(base));
    base2node_it[base] = nodes.begin();
}
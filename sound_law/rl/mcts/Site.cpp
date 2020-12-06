#include "Site.hpp"
#include "Word.hpp"

SiteNode::SiteNode(const Site &site) : site(site) {}

void SiteNode::debug()
{
    std::cerr << "Debug Site:\n";
    for (abc_t i : site)
        std::cerr << i << '\n';
}

SiteNode *SiteSpace::get_node(abc_t before_id,
                              abc_t pre_id,
                              abc_t d_pre_id,
                              abc_t post_id,
                              abc_t d_post_id)
{
    // Skip generation if this site has already been seen.
    Site site = Site{before_id, pre_id, d_pre_id, post_id, d_post_id};
    // Obtain the read lock for membership test.
    nodes_mtx.lock_shared();
    if (nodes.find(site) != nodes.end())
    {
        SiteNode *node = nodes.at(site);
        nodes_mtx.unlock_shared();
        return node;
    }
    nodes_mtx.unlock_shared();

    // Recursively generate the subgraph.
    SiteNode *new_node = new SiteNode(site);
    // Obtain the write lock.
    nodes_mtx.lock();
    if (nodes.find(site) == nodes.end())
        nodes[site] = new_node;
    else
    {
        // Release the memory and stop the recursion.
        delete new_node;
        new_node = nodes.at(site);
        nodes_mtx.unlock();
        return new_node;
    }
    nodes_mtx.unlock();
    if (pre_id != NULL_abc)
    {
        if (d_pre_id != NULL_abc)
            new_node->lchild = get_node(before_id, pre_id, NULL_abc, post_id, d_post_id);
        new_node->lchild = get_node(before_id, NULL_abc, NULL_abc, post_id, d_post_id);
    }
    if (post_id != NULL_abc)
    {
        if (d_post_id != NULL_abc)
            new_node->rchild = get_node(before_id, pre_id, d_pre_id, post_id, NULL_abc);

        new_node->rchild = get_node(before_id, pre_id, d_pre_id, NULL_abc, NULL_abc);
    }
    return new_node;
}

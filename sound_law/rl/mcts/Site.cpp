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
    {
        boost::shared_lock_guard<boost::shared_mutex> lock(nodes_mtx);
        if (nodes.find(site) != nodes.end())
            return nodes.at(site);
    }

    // Recursively generate the subgraph.
    SiteNode *new_node = new SiteNode(site);
    // Obtain the write lock.
    {
        boost::lock_guard<boost::shared_mutex> lock(nodes_mtx);
        if (nodes.find(site) == nodes.end())
            nodes[site] = new_node;
        else
        {
            // Release the memory and stop the recursion.
            delete new_node;
            return nodes.at(site);
        }
    }
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

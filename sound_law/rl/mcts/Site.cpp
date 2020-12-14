#include "Site.hpp"
#include "Word.hpp"

SiteNode::SiteNode(usi_t site) : site(site) {}

SiteSpace::SiteSpace(abc_t sot_id,
                     abc_t eot_id,
                     abc_t any_id,
                     abc_t emp_id) : sot_id(sot_id),
                                     eot_id(eot_id),
                                     any_id(any_id),
                                     emp_id(emp_id) {}

SiteNode *SiteSpace::get_node(abc_t before_id,
                              abc_t pre_id,
                              abc_t d_pre_id,
                              abc_t post_id,
                              abc_t d_post_id)
{
    // Skip generation if this site has already been seen.
    // Obtain the read lock for membership test.
    usi_t site = site::combine(pre_id, d_pre_id, post_id, d_post_id, before_id);
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
    if (pre_id != NULL_ABC)
    {
        if (d_pre_id != NULL_ABC)
        {
            if ((d_pre_id != any_id) && (d_pre_id != sot_id))
                new_node->lxchild = get_node(before_id, pre_id, any_id, post_id, d_post_id);
            new_node->lchild = get_node(before_id, pre_id, NULL_ABC, post_id, d_post_id);
        }
        else
        {
            if ((pre_id != any_id) && (pre_id != sot_id))
                new_node->lxchild = get_node(before_id, any_id, NULL_ABC, post_id, d_post_id);
            new_node->lchild = get_node(before_id, NULL_ABC, NULL_ABC, post_id, d_post_id);
        }
    }
    if (post_id != NULL_ABC)
    {
        if (d_post_id != NULL_ABC)
        {
            if ((d_post_id != any_id) && (d_post_id != eot_id))
                new_node->rxchild = get_node(before_id, pre_id, d_pre_id, post_id, any_id);
            new_node->rchild = get_node(before_id, pre_id, d_pre_id, post_id, NULL_ABC);
        }
        else
        {
            if ((post_id != any_id) && (post_id != eot_id))
                new_node->rxchild = get_node(before_id, pre_id, d_pre_id, any_id, NULL_ABC);
            new_node->rchild = get_node(before_id, pre_id, d_pre_id, NULL_ABC, NULL_ABC);
        }
    }
    return new_node;
}

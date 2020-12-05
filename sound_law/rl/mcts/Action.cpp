#include "Action.hpp"
#include "TreeNode.hpp"

ActionSpace::ActionSpace(SiteSpace *site_space) : site_space(site_space) {}

void ActionSpace::register_edges(abc_t before_id, abc_t after_id)
{
    edges[before_id].push_back(after_id);
}

action_t ActionSpace::get_action_id(const Action &action)
{
    if (a2i.find(action) != a2i.end())
        return a2i.at(action);

    action_t action_id = (action_t)a2i.size();
    a2i[action] = action_id;
    return action_id;
}

void ActionSpace::set_action_allowed(TreeNode *t_node)
{
    assert(!t_node->action_allowed.empty());
    std::vector<action_t> &aa = t_node->action_allowed;

    // Build the graph first.
    SiteGraph graph = SiteGraph(site_space);
    for (Word *word : t_node->words)
        for (SiteNode *root : word->site_roots)
            graph.add_root(root);

    for (const auto &item : graph.nodes)
    {
        GraphNode *g_node = item.second;
        if ((g_node->lchild != nullptr) && (g_node->lchild->num_sites == g_node->num_sites))
            continue;
        if ((g_node->rchild != nullptr) && (g_node->rchild->num_sites == g_node->num_sites))
            continue;
        const Site &site = g_node->base->site;
        abc_t before_id = site.at(0);
        abc_t pre_id = site.at(1);
        abc_t d_pre_id = site.at(2);
        abc_t post_id = site.at(3);
        abc_t d_post_id = site.at(4);
        for (abc_t after_id : edges.at(before_id))
        {
            Action action = Action{before_id, after_id, pre_id, d_pre_id, post_id, d_post_id};
            action_t action_id = get_action_id(action);
            aa.push_back(action_id);
        }
    }
    assert(!aa.empty());
}
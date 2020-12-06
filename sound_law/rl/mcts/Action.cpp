#include "Action.hpp"
#include "TreeNode.hpp"

ActionSpace::ActionSpace(
    SiteSpace *site_space,
    WordSpace *word_space) : site_space(site_space),
                             word_space(word_space)
{
    // Reserve the first action for the stop action.
    Action action = Action{NULL_abc, NULL_abc, NULL_abc, NULL_abc, NULL_abc, NULL_abc};
    actions.push_back(action);
    a2i[action] = 0;
}

void ActionSpace::register_edge(abc_t before_id, abc_t after_id)
{
    edges[before_id].push_back(after_id);
}

action_t ActionSpace::safe_get_action_id(const Action &action)
{
    actions_mtx.lock_shared();
    if (a2i.find(action) != a2i.end())
    {
        action_t action_id = a2i.at(action);
        actions_mtx.unlock_shared();
        return action_id;
    }
    actions_mtx.unlock_shared();

    actions_mtx.lock();
    action_t action_id = (action_t)a2i.size();
    a2i[action] = action_id;
    actions.push_back(action);
    actions_mtx.unlock();
    return action_id;
}

action_t ActionSpace::get_action_id(const Action &action) { return a2i.at(action); }

Action ActionSpace::get_action(action_t action_id) { return actions.at(action_id); }

void ActionSpace::set_action_allowed(TreeNode *t_node)
{
    // Skip this if actions have already been set before.
    if (!t_node->action_allowed.empty())
        return;

    std::vector<action_t> &aa = t_node->action_allowed;
    // Stop action is almost available.
    aa.push_back(0);

    // Build the graph first.
    SiteGraph graph = SiteGraph(site_space);
    for (int i = 0; i < t_node->words.size(); ++i)
    {
        Word *word = t_node->words.at(i);
        for (SiteNode *root : word->site_roots)
            graph.add_root(root, i);
    }

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
            action_t action_id = safe_get_action_id(action);
            float delta = 0.0;
            for (int order : g_node->linked_words)
            {
                Word *word = t_node->words.at(order);
                Word *new_word = word_space->apply_action(word, action, order);
                delta += new_word->dist - word->dist;
            }

            if (delta < 0.0)
                aa.push_back(action_id);
        }
    }
    assert(!aa.empty());
}
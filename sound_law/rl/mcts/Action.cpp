#include "Action.hpp"
#include "TreeNode.hpp"

ActionSpace::ActionSpace(
    SiteSpace *site_space,
    WordSpace *word_space,
    int num_threads) : site_space(site_space),
                       word_space(word_space)
{
    // Reserve the first action for the stop action.
    Action action = Action{NULL_abc, NULL_abc, NULL_abc, NULL_abc, NULL_abc, NULL_abc};
    actions.push_back(action);
    a2i[action] = 0;
    a2i_cache.push_back(NULL_abc);
    a2i_cache.push_back(NULL_abc);
    a2i_cache.push_back(NULL_abc);
    a2i_cache.push_back(NULL_abc);
    a2i_cache.push_back(NULL_abc);
    a2i_cache.push_back(NULL_abc);

    // Initialize thread pool.
    tp.resize(num_threads);
}

void ActionSpace::register_edge(abc_t before_id, abc_t after_id)
{
    edges[before_id].push_back(after_id);
}

action_t ActionSpace::safe_get_action_id(const Action &action)
{
    {
        boost::shared_lock_guard<boost::shared_mutex> lock(actions_mtx);
        if (a2i.find(action) != a2i.end())
            return a2i.at(action);
    }

    boost::lock_guard<boost::shared_mutex> lock(actions_mtx);
    action_t action_id = (action_t)a2i.size();
    a2i[action] = action_id;
    actions.push_back(action);
    a2i_cache.push_back(action.at(0));
    a2i_cache.push_back(action.at(1));
    a2i_cache.push_back(action.at(2));
    a2i_cache.push_back(action.at(3));
    a2i_cache.push_back(action.at(4));
    a2i_cache.push_back(action.at(5));
    return action_id;
}

action_t ActionSpace::get_action_id(const Action &action) { return a2i.at(action); }

Action ActionSpace::get_action(action_t action_id) { return actions.at(action_id); }

void ActionSpace::set_action_allowed(TreeNode *t_node)
{
    // Skip this if it has been stopped.
    if (t_node->stopped)
        return;

    boost::lock_guard<boost::mutex> lock(t_node->exclusive_mtx);
    set_action_allowed_no_lock(t_node);
}

void ActionSpace::set_action_allowed_no_lock(TreeNode *t_node)
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
                Word *new_word = word_space->apply_action(word, action_id, action, order);
                delta += new_word->dist - word->dist;
            }

            if (delta < 0.0)
                aa.push_back(action_id);
        }
    }
    assert(!aa.empty());
}

void ActionSpace::set_action_allowed(const std::vector<TreeNode *> &nodes)
{
    // Filter out the duplicates.
    auto unique_idx = boost::unordered_set<tn_cnt_t>();
    auto unique_nodes = std::vector<TreeNode *>();
    for (const auto node : nodes)
        if (unique_idx.find(node->idx) == unique_idx.end())
        {
            unique_idx.insert(node->idx);
            unique_nodes.push_back(node);
        }

    // Since they are all unique, we can be lock-free.
    std::vector<std::future<void>> results(unique_nodes.size());
    for (int i = 0; i < unique_nodes.size(); i++)
    {
        auto node = unique_nodes.at(i);
        results[i] = tp.push([this, node](int id) { this->set_action_allowed_no_lock(node); });
    }
    for (int i = 0; i < unique_nodes.size(); i++)
        results[i].get();
}

size_t ActionSpace::size() { return actions.size(); }

std::vector<abc_t> ActionSpace::expand_a2i()
{
    std::vector<abc_t> ret = a2i_cache;
    a2i_cache.clear();
    return ret;
}
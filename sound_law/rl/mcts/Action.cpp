#include "Action.hpp"
#include "TreeNode.hpp"

ActionSpace::ActionSpace(
    SiteSpace *site_space,
    WordSpace *word_space,
    int num_threads) : site_space(site_space),
                       word_space(word_space)
{
    // Initialize thread pool.
    tp.resize(num_threads);
}

void ActionSpace::register_edge(abc_t before_id, abc_t after_id)
{
    edges[before_id].push_back(after_id);
}

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

    std::vector<uai_t> &aa = t_node->action_allowed;
    // Stop action is almost available.
    aa.push_back(action::STOP);

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
        usi_t site = g_node->base->site;
        abc_t before_id = site::get_before_id(site);
        for (abc_t after_id : edges.at(before_id))
        {
            uai_t action_id = action::combine_after_id(site, after_id);
            float delta = 0.0;
            for (int order : g_node->linked_words)
            {
                Word *word = t_node->words.at(order);
                Word *new_word = word_space->apply_action(word, action_id, order);
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
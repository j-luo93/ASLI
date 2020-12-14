#include "Action.hpp"
#include "TreeNode.hpp"

ActionSpace::ActionSpace(
    SiteSpace *site_space,
    WordSpace *word_space,
    float prune_threshold,
    int num_threads) : site_space(site_space),
                       word_space(word_space),
                       prune_threshold(prune_threshold)
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

inline void set_affected_action_words_pairs(
    std::vector<uai_t> &unseen_actions,
    std::vector<std::vector<int>> &unseen_words,
    TreeNode *t_node,
    SiteSpace *site_space,
    const UMap<abc_t, std::vector<abc_t>> &edges)
{
    // Build the graph first.
    SiteGraph graph = SiteGraph(site_space);
    for (int i = 0; i < t_node->words.size(); ++i)
    {
        Word *word = t_node->words.at(i);
        for (SiteNode *root : word->site_roots)
            graph.add_root(root, i);
    }
    // Get all unseen action-words pairs.
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
            unseen_actions.push_back(action_id);
            unseen_words.push_back(std::vector<int>(g_node->linked_words.begin(),
                                                    g_node->linked_words.end()));
        }
    }
}

void ActionSpace::find_potential_actions(TreeNode *t_node,
                                         std::vector<uai_t> &potential_actions,
                                         std::vector<std::vector<int>> &potential_orders,
                                         std::vector<uai_t> &unseen_actions,
                                         std::vector<std::vector<int>> &unseen_orders)
{
    // Build the graph first.
    SiteGraph graph = SiteGraph(site_space);
    for (int i = 0; i < t_node->words.size(); ++i)
    {
        Word *word = t_node->words.at(i);
        for (SiteNode *root : word->site_roots)
            graph.add_root(root, i);
    }
    // Get all unseen action-words pairs.
    for (const auto &item : graph.nodes)
    {
        GraphNode *g_node = item.second;
        if ((g_node->lchild != nullptr) && (g_node->lchild->num_sites == g_node->num_sites))
            continue;
        if ((g_node->lxchild != nullptr) && (g_node->lxchild->num_sites == g_node->num_sites))
            continue;
        if ((g_node->rchild != nullptr) && (g_node->rchild->num_sites == g_node->num_sites))
            continue;
        if ((g_node->rxchild != nullptr) && (g_node->rxchild->num_sites == g_node->num_sites))
            continue;

        usi_t site = g_node->base->site;
        abc_t before_id = site::get_before_id(site);
        // FIXME(j_luo) This can be further optimized.
        for (abc_t after_id : edges.at(before_id))
        {
            uai_t action_id = action::combine_after_id(site, after_id);
            potential_actions.push_back(action_id);
            auto po = std::vector<int>();
            auto uo = std::vector<int>();
            for (auto order : g_node->linked_words)
            {
                auto word = t_node->words.at(order);
                po.push_back(order);
                if (word->neighbors.find(action_id) == word->neighbors.end())
                    uo.push_back(order);
            }
            potential_orders.push_back(po);
            if (!uo.empty())
            {
                unseen_actions.push_back(action_id);
                unseen_orders.push_back(uo);
            }
        }
    }
}

void ActionSpace::set_action_allowed_no_lock(TreeNode *t_node)
{
    // Skip this if actions have already been set before.
    if (!t_node->action_allowed.empty())
        return;

    std::vector<uai_t> &aa = t_node->action_allowed;
    // Stop action is always available.
    aa.push_back(action::STOP);

    // Get the graph first.
    auto unseen_actions = std::vector<uai_t>();
    auto unseen_words = std::vector<std::vector<int>>();
    set_affected_action_words_pairs(unseen_actions, unseen_words, t_node, site_space, edges);
    for (size_t i = 0; i < unseen_words.size(); i++)
    {
        uai_t action_id = unseen_actions.at(i);
        bool epenthesis = (action::get_after_id(action_id) == site_space->emp_id);
        const auto &words = unseen_words.at(i);
        float delta = 0.0;
        for (const auto order : words)
        {
            Word *word = t_node->words.at(order);
            // Cannot delete anything if the word has only one character (other than SOT and EOT).
            if (epenthesis && (word->size() == 3))
            {
                delta = 9999999999.9;
                break;
            }
            Word *new_word = word_space->apply_action(word, action_id, order);
            delta += new_word->dist - word->dist;
        }

        if (delta < prune_threshold)
            aa.push_back(action_id);
    }
    assert(!aa.empty());
}

inline void get_unseen_neighbors(std::vector<Word *> &unseen_words,
                                 std::vector<uai_t> &unseen_actions,
                                 SiteGraph *graph,
                                 TreeNode *t_node,
                                 const UMap<abc_t, std::vector<abc_t>> &edges)
{
    for (const auto &item : graph->nodes)
    {
        auto *g_node = item.second;
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
                if (word->neighbors.find(action_id) == word->neighbors.end())
                {
                    unseen_words.push_back(word);
                    unseen_actions.push_back(action_id);
                }
            }
        }
    }
}
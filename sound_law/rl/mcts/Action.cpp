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
    set_action_allowed(std::vector<TreeNode *>{t_node});
}

void ActionSpace::set_action_allowed(const std::vector<TreeNode *> &nodes)
{
    // Get the unique set of nodes to set action allowed for.
    auto unique_nodes = std::vector<TreeNode *>();
    auto unique_idx = boost::unordered_set<tn_cnt_t>();
    for (const auto node : nodes)
        if ((!node->stopped) && (!node->done) && (node->action_allowed.empty()) && (unique_idx.find(node->idx) == unique_idx.end()))
        {
            unique_idx.insert(node->idx);
            unique_nodes.push_back(node);
        }
    size_t num_nodes = unique_nodes.size();
    std::vector<std::future<void>> p_results(num_nodes);

    // Get potential and unseen action-orders pairs.
    auto para_potential_actions = std::vector<std::vector<uai_t>>(num_nodes);
    auto para_potential_orders = std::vector<std::vector<std::vector<int>>>(num_nodes);
    auto para_unseen_actions = std::vector<std::vector<uai_t>>(num_nodes);
    auto para_unseen_orders = std::vector<std::vector<std::vector<int>>>(num_nodes);

    for (int i = 0; i < num_nodes; i++)
    {
        auto node = unique_nodes.at(i);
        auto &pa = para_potential_actions[i];
        auto &po = para_potential_orders[i];
        auto &ua = para_unseen_actions[i];
        auto &uo = para_unseen_orders[i];
        p_results[i] = tp.push([this, node, &pa, &po, &ua, &uo](int) { this->find_potential_actions(node, pa, po, ua, uo); });
    }
    for (int i = 0; i < num_nodes; i++)
        p_results[i].get();

    // Gather all unique unseen applications.
    auto unique_unseen_actions = boost::unordered_map<Word *, boost::unordered_map<int, boost::unordered_set<uai_t>>>();
    for (int i = 0; i < num_nodes; i++)
    {
        auto node = unique_nodes.at(i);
        auto &ua = para_unseen_actions.at(i);
        auto &uo = para_unseen_orders.at(i);
        for (int j = 0; j < ua.size(); j++)
        {
            auto action_id = ua.at(j);
            auto &inner_uo = uo.at(j);
            for (const auto order : inner_uo)
            {
                auto word = node->words.at(order);
                unique_unseen_actions[word][order].insert(action_id);
            }
        }
    }

    auto unseen_words = std::vector<Word *>();
    auto unseen_orders = std::vector<std::vector<int>>();
    auto unseen_actions = std::vector<std::vector<std::vector<uai_t>>>();
    for (const auto &item : unique_unseen_actions)
    {
        unseen_words.emplace_back(item.first);
        unseen_orders.emplace_back(std::vector<int>());
        unseen_actions.emplace_back(std::vector<std::vector<uai_t>>());
        auto &uo = unseen_orders.back();
        auto &ua = unseen_actions.back();
        for (const auto &inner_item : item.second)
        {
            uo.emplace_back(inner_item.first);
            ua.emplace_back(std::vector<uai_t>(inner_item.second.begin(), inner_item.second.end()));
        }
    }

    // Add the new neighbors.
    size_t num_words = unseen_words.size();
    std::vector<std::future<void>> w_results(num_words);
    for (int i = 0; i < num_words; i++)
    {
        auto word = unseen_words.at(i);
        auto &uo = unseen_orders.at(i);
        auto &ua = unseen_actions.at(i);
        w_results[i] = tp.push([this, word, &uo, &ua](int) { this->apply_new_word_no_lock(word, uo, ua); });
    }
    for (int i = 0; i < num_words; i++)
        w_results[i].get();

    // Actually set values to everything.
    std::vector<std::future<void>> a_results(num_nodes);
    for (int i = 0; i < num_nodes; i++)
    {
        auto node = unique_nodes.at(i);
        auto &pa = para_potential_actions.at(i);
        auto &po = para_potential_orders.at(i);
        a_results[i] = tp.push([this, node, &pa, &po](int) { set_action_allowed_no_lock(node, pa, po); });
    }
    for (int i = 0; i < num_nodes; i++)
        a_results[i].get();
}

void ActionSpace::apply_new_word_no_lock(Word *word,
                                         const std::vector<int> &unseen_orders,
                                         const std::vector<std::vector<uai_t>> &unseen_actions)
{

    for (int i = 0; i < unseen_orders.size(); i++)
    {
        auto order = unseen_orders.at(i);
        auto &ua = unseen_actions.at(i);
        for (auto action_id : ua)
        {
            auto new_word = word_space->apply_action_no_lock(word, action_id, order);
            word->neighbors[action_id] = new_word;
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

void ActionSpace::set_action_allowed_no_lock(TreeNode *node,
                                             const std::vector<uai_t> &potential_actions,
                                             const std::vector<std::vector<int>> &potential_orders)
{
    auto &aa = node->action_allowed;
    // Stop is always available.
    aa.push_back(action::STOP);
    for (int j = 0; j < potential_actions.size(); j++)
    {
        auto action_id = potential_actions.at(j);
        bool epenthesis = (action::get_after_id(action_id) == site_space->emp_id);
        auto &po = potential_orders.at(j);
        float delta = 0.0;
        for (auto order : po)
        {
            auto word = node->words.at(order);
            if (epenthesis && (word->size() == 3))
            {
                delta = 9999999999.9;
                break;
            }
            auto new_word = word->neighbors.at(action_id);
            delta += new_word->dists.at(order) - word->dists.at(order);
        }
        if (delta < prune_threshold)
        {
            aa.push_back(action_id);
        }
    }
    assert(!aa.empty());
}

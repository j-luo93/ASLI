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
        const auto &words = unseen_words.at(i);
        float delta = 0.0;
        for (const auto order : words)
        {
            Word *word = t_node->words.at(order);
            Word *new_word = word_space->apply_action(word, action_id, order);
            delta += new_word->dist - word->dist;
        }

        if (delta < 0.0)
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

void ActionSpace::set_action_allowed(const std::vector<TreeNode *> &nodes)
{
    // Filter out the duplicates and those already explored.
    auto unique_idx = boost::unordered_set<tn_cnt_t>();
    auto unique_nodes = std::vector<TreeNode *>();
    for (const auto node : nodes)
        if ((node->action_allowed.empty()) && (unique_idx.find(node->idx) == unique_idx.end()))
        {
            unique_idx.insert(node->idx);
            unique_nodes.push_back(node);
        }

    // Since they are all unique, we can be lock-free.
    size_t num_jobs = unique_nodes.size();
    auto results = std::vector<std::future<void>>(num_jobs);
    //Build the graphs first.
    auto graphs = std::vector<SiteGraph *>(num_jobs);
    // for (int job_idx = 0; job_idx < num_jobs; job_idx++)
    // {
    //     auto node = unique_nodes.at(job_idx);
    //     results[job_idx] = tp.push([this, node, job_idx, &graphs](int id) { graphs[job_idx] = this->get_graph(node); });
    // }
    for (int job_idx = 0; job_idx < num_jobs; job_idx++)
        results[job_idx].get();
    // Get unseen word-action pairs.
    auto unseen_words = std::vector<std::vector<Word *>>(num_jobs);
    auto unseen_actions = std::vector<std::vector<uai_t>>(num_jobs);
    for (int job_idx = 0; job_idx < num_jobs; job_idx++)
    {
        auto node = unique_nodes.at(job_idx);
        // Use this hack to get around the weird lambda constraint: https://stackoverflow.com/questions/32922053/c-lambda-capture-member-variable.
        auto &edges = this->edges;
        results[job_idx] = tp.push([node, job_idx,
                                    &edges,
                                    &unseen_words,
                                    &unseen_actions,
                                    &graphs](int id) { get_unseen_neighbors(unseen_words[job_idx],
                                                                            unseen_actions[job_idx],
                                                                            graphs[job_idx],
                                                                            node, edges); });
    }
    for (int job_idx = 0; job_idx < num_jobs; job_idx++)
        results[job_idx].get();

    // for (int i = 0; i < unique_nodes.size(); i++)
    // {
    //     auto node = unique_nodes.at(i);
    //     results[i] = tp.push([this, node](int id) { this->set_action_allowed_no_lock(node); });
    // }
    // for (int i = 0; i < unique_nodes.size(); i++)
    //     results[i].get();
}
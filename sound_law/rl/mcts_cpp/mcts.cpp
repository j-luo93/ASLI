#include "mcts.hpp"

Mcts::Mcts(Env *env, const MctsOpt &opt) : env(env), opt(opt)
{
    if (opt.num_threads > 1)
        tp = new Pool(opt.num_threads);
    else
        tp = nullptr;
}

TreeNode *Mcts::select_single_thread(TreeNode *node, int depth_limit)
{
    assert(!node->is_leaf());
    while ((node->depth < depth_limit) && (!node->is_leaf()))
    {
        // Complete sampling one action.
        SPDLOG_DEBUG("Mcts: node depth {}", node->depth);
        SPDLOG_DEBUG("Mcts: node str\n{}", str::from(node));
        auto subpath = env->action_space->get_best_subpath(node, opt.puct_c, opt.game_count, opt.virtual_loss);
        SPDLOG_DEBUG("Mcts: node subpath found.");
        node = env->apply_action(node, subpath);
        SPDLOG_DEBUG("Mcts: action applied.");
        if ((node->stopped) || (node->done))
            break;
    }
    SPDLOG_DEBUG("Mcts: selected node dist {0} str\n{1}", node->dist, str::from(node));
    return node;
}

vec<TreeNode *> Mcts::select(TreeNode *root, int num_sims, int depth_limit)
{
    SPDLOG_DEBUG("Mcts: selecting...");
    auto selected = vec<TreeNode *>();
    selected.reserve(num_sims);
    if (tp == nullptr)
        for (size_t i = 0; i < num_sims; ++i)
            selected.push_back(select_single_thread(root, depth_limit));
    else
    {
        vec<std::future<void>> results(num_sims);
        selected.resize(num_sims);
        for (size_t i = 0; i < num_sims; ++i)
            results[i] = tp->push(
                [this, root, depth_limit, i, &selected](int) {
                    selected[i] = this->select_single_thread(root, depth_limit);
                });
        for (size_t i = 0; i < num_sims; ++i)
            results[i].wait();
    }
    SPDLOG_DEBUG("Mcts: selected.");
    return selected;
}

void Mcts::backup(const vec<TreeNode *> &nodes, const vec<float> &values)
{
    assert(nodes.size() == values.size());
    for (size_t i = 0; i < nodes.size(); i++)
    {
        auto node = nodes[i];
        auto value = values[i];
        node->backup(value, opt.game_count, opt.virtual_loss);
    }
}
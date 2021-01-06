#include "mcts.hpp"

Mcts::Mcts(
    Env *env,
    float puct_c,
    int game_count,
    float virtual_loss,
    int num_threads) : env(env),
                       puct_c(puct_c),
                       game_count(game_count),
                       virtual_loss(virtual_loss),
                       num_threads(num_threads)
{
    if (num_threads > 1)
        tp = new Pool(num_threads);
    else
        tp = nullptr;

    env->action_space->set_action_allowed(tp, vec<TreeNode *>{env->start});
}

vec<TreeNode *> Mcts::select(TreeNode *root, int num_sims, int depth_limit)
{
    auto selected = vec<TreeNode *>();
    parallel_apply<false, 1>(
        tp,
        [this, depth_limit](TreeNode *node) {
            SPDLOG_DEBUG("Starting selection.");
            assert(!node->is_leaf());
            SPDLOG_DEBUG("Node depth: {}", node->depth);
            while ((node->depth < depth_limit) && (!node->is_leaf()))
            {
                SPDLOG_DEBUG("Current node: {}", node->str());
                SPDLOG_DEBUG(node->str());
                auto best_i = node->select(puct_c, game_count, virtual_loss);
                auto action = node->action_allowed.at(best_i);
                node = env->apply_action(node, best_i, action);
                SPDLOG_DEBUG("Selected best_i {0} action {1}. New node: {2}", best_i, action::str(action), node->str());
                SPDLOG_DEBUG(node->str());
                if ((node->stopped) || (node->done))
                    break;
            }
            SPDLOG_DEBUG("Selection finished.");
            return node;
        },
        selected,
        vec<TreeNode *>(num_sims, root));
    env->action_space->set_action_allowed(tp, selected);
    return selected;
}

void Mcts::backup(const vec<TreeNode *> &nodes, const vec<float> &values)
{
    assert(nodes.size() == values.size());
    for (size_t i = 0; i < nodes.size(); i++)
    {
        auto node = nodes.at(i);
        auto value = values.at(i);
        node->backup(value, game_count, virtual_loss);
    }
}

uai_t Mcts::play(TreeNode *node)
{
    assert(!node->played);
    node->played = true;
    int best_i = node->max_index;
    uai_t action_id = (best_i == -1) ? action::STOP : node->max_action_id;
    SPDLOG_INFO("Played action {0}, best_i {1}", action::str(action_id), best_i);
    SPDLOG_INFO("Old node: {}", node->str());
    SPDLOG_INFO("New node: {}", node->neighbors.at(action_id)->str());
    return action_id;
}

void Mcts::set_logging_options(int verbose_level, bool log_to_file = false)
{
    spdlog::level::level_enum level;
    switch (verbose_level)
    {
    case 0:
        level = spdlog::level::err;
        break;
    case 1:
        level = spdlog::level::info;
        break;
    case 2:
        level = spdlog::level::debug;
        break;
    case 3:
        level = spdlog::level::trace;
        break;
    }
    spdlog::set_level(level);

    if (log_to_file)
    {
        auto logger = spdlog::basic_logger_st("default", "log.txt", true);
        logger->flush_on(level);
        spdlog::set_default_logger(logger);
    }
}
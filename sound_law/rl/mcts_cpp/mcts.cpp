#include "mcts.hpp"

vec<Edge> Path::get_edges_to_root() const
{
    assert(subpaths.size() == tree_nodes.size() - 1);
    auto edges = vec<Edge>();
    edges.reserve(subpaths.size() * 7 + 1);
    int i = tree_nodes.size() - 1;
    for (auto it = subpaths.crbegin(); it != subpaths.crend(); ++it)
    {
        auto &subpath = *it;
        auto edge = Edge();
        edge.s0 = tree_nodes[i];
        edge.a = subpath.chosen_seq[6];
        edge.s1 = subpath.mini_node_seq[5];
        edges.push_back(edge);

        for (int j = 5; j >= 1; --j)
        {
            edge = Edge();
            edge.s0 = subpath.mini_node_seq[j];
            edge.a = subpath.chosen_seq[j];
            edge.s1 = subpath.mini_node_seq[j - 1];
            edges.push_back(edge);
        }

        edge = Edge();
        edge.s0 = subpath.mini_node_seq[0];
        edge.a = subpath.chosen_seq[0];
        assert(i >= 1);
        edge.s1 = tree_nodes[i - 1];
        edges.push_back(edge);

        --i;
    }
    return edges;
}

Mcts::Mcts(Env *env, const MctsOpt &opt) : env(env), opt(opt)
{
    if (opt.num_threads > 1)
        tp = new Pool(opt.num_threads);
    else
        tp = nullptr;
}

Path Mcts::select_single_thread(TreeNode *node, int depth_limit) const
{
    assert(!node->is_leaf());
    auto path = Path();
    path.tree_nodes.push_back(node);
    while ((node->depth < depth_limit) && (!node->is_leaf()))
    {
        // Complete sampling one action.
        SPDLOG_DEBUG("Mcts: node depth {}", node->depth);
        SPDLOG_DEBUG("Mcts: node str\n{}", str::from(node));
        auto subpath = env->action_space->get_best_subpath(node, opt.puct_c, opt.game_count, opt.virtual_loss, opt.heur_c);
        SPDLOG_DEBUG("Mcts: node subpath found.");
        node = env->apply_action(node, subpath);
        path.subpaths.push_back(subpath);
        path.tree_nodes.push_back(node);
        SPDLOG_DEBUG("Mcts: action applied.");
        if ((node->stopped) || (node->done))
            break;
    }
    SPDLOG_DEBUG("Mcts: selected node dist {0} str\n{1}", node->dist, str::from(node));
    return path;
}

vec<Path> Mcts::select(TreeNode *root, int num_sims, int depth_limit) const
{
    SPDLOG_DEBUG("Mcts: selecting...");
    auto paths = vec<Path>();
    paths.reserve(num_sims);
    if (tp == nullptr)
        for (size_t i = 0; i < num_sims; ++i)
            paths.push_back(select_single_thread(root, depth_limit));
    else
    {
        vec<std::future<void>> results(num_sims);
        paths.resize(num_sims);
        for (size_t i = 0; i < num_sims; ++i)
            results[i] = tp->push(
                [this, root, depth_limit, i, &paths](int) {
                    paths[i] = this->select_single_thread(root, depth_limit);
                });
        for (size_t i = 0; i < num_sims; ++i)
            results[i].wait();
    }
    SPDLOG_DEBUG("Mcts: selected.");
    return paths;
}

void Mcts::backup(const vec<Path> &paths, const vec<float> &values) const
{
    assert(paths.size() == values.size());
    for (size_t i = 0; i < paths.size(); i++)
    {
        auto value = values[i];
        auto edges = paths[i].get_edges_to_root();

        float rtg = 0.0;
        for (auto it = edges.begin(); it != edges.end(); ++it)
        {
            auto &edge = *it;
            int index = edge.a.first;
            abc_t best_char = edge.a.second;
            BaseNode *parent = edge.s1; // Since the edge points from child to parent, `s1` is used instead of `s0`.
            // auto tparent = dynamic_cast<TransitionNode *>(parent);
            // if (tparent != nullptr)
            //     rtg += tparent->rewards[index];
            if (parent->is_transitional())
                rtg += static_cast<TransitionNode *>(parent)->rewards[index];
            parent->action_counts[index] -= opt.game_count - 1;
            if (parent->action_counts[index] < 1)
            {
                std::cerr << index << '\n';
                std::cerr << best_char << '\n';
                std::cerr << parent->action_counts[index] << '\n';
                assert(false);
            }
            // Update max value of the parent.
            float new_value = value + rtg;
            if (new_value > parent->max_value)
            {
                parent->max_value = new_value;
                parent->max_index = index;
            }
            if (new_value > parent->max_values[index])
                parent->max_values[index] = new_value;
            parent->total_values[index] += opt.game_count * opt.virtual_loss + new_value;
            parent->visit_count -= opt.game_count - 1;
        }
    }
}
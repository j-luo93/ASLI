#include "mcts.hpp"

Path::Path(TreeNode *start, const int start_depth) : depth(start_depth) { tree_nodes.push_back(start); }

int Path::get_depth() const { return depth; }

void Path::append(const Subpath &subpath, TreeNode *node)
{
    subpaths.push_back(subpath);
    tree_nodes.push_back(node);
    ++depth;
}

bool Path::forms_a_circle(TreeNode *node) const
{
    return std::find(tree_nodes.begin(), tree_nodes.end(), node) != tree_nodes.end();
}

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

Path Mcts::select_single_thread(TreeNode *node, const int start_depth, const int depth_limit, const Path &old_path) const
{
    assert(!node->is_leaf());
    auto path = Path(old_path);              // This extends the old path. Used for detecting circles.
    auto new_path = Path(node, start_depth); // This only records the extended part. Used for backing up values later.
    while ((new_path.get_depth() < depth_limit) && (!node->is_leaf()))
    {
        // Complete sampling one action.
        SPDLOG_DEBUG("Mcts: node str\n{}", str::from(node));
        auto subpath = env->action_space->get_best_subpath(node, opt.puct_c, opt.game_count, opt.virtual_loss, opt.heur_c, opt.add_noise, opt.use_num_misaligned, opt.use_max_value);
        SPDLOG_DEBUG("Mcts: node subpath found.");

        // Add virtual loss.
        StatsManager::virtual_select(node, subpath.chosen_seq[0].first, opt.game_count, opt.virtual_loss);
        for (size_t i = 0; i < 6; ++i)
            StatsManager::virtual_select(subpath.mini_node_seq[i], subpath.chosen_seq[i + 1].first, opt.game_count, opt.virtual_loss);

        node = env->apply_action(node, subpath);
        bool is_circle = path.forms_a_circle(node);
        if (is_circle)
        {
            SPDLOG_DEBUG("Mcts: found a circle at {}!", str::from(node));
            PruningManager::prune(subpath.mini_node_seq[5], subpath.chosen_seq[6].first);
        }
        new_path.append(subpath, node);
        path.append(subpath, node);
        SPDLOG_DEBUG("Mcts: action applied.");
        if ((node->stopped) || (node->is_done()) || (is_circle))
            break;
    }
    SPDLOG_DEBUG("Mcts: selected node dist {0} str\n{1}", node->get_dist(), str::from(node));
    return new_path;
}

vec<Path> Mcts::select(TreeNode *root, const int num_sims, const int start_depth, const int depth_limit) const
{
    assert(start_depth == 0);
    auto old_path = Path(root, 0);
    return select(root, num_sims, start_depth, depth_limit, old_path);
}

vec<Path> Mcts::select(TreeNode *root, const int num_sims, const int start_depth, const int depth_limit, const Path &old_path) const
{
    SPDLOG_DEBUG("Mcts: selecting...");
    auto paths = vec<Path>();
    paths.reserve(num_sims);
    if (tp == nullptr)
        for (size_t i = 0; i < num_sims; ++i)
            paths.push_back(select_single_thread(root, start_depth, depth_limit, old_path));
    else
    {
        vec<std::future<void>> results(num_sims);
        paths.resize(num_sims);
        for (size_t i = 0; i < num_sims; ++i)
            results[i] = tp->push(
                [this, root, start_depth, depth_limit, i, &old_path, &paths](int) {
                    paths[i] = this->select_single_thread(root, start_depth, depth_limit, old_path);
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
            // abc_t best_char = edge.a.second;
            BaseNode *parent = edge.s1; // Since the edge points from child to parent, `s1` is used instead of `s0`.
            if (parent->is_transitional())
                rtg += static_cast<TransitionNode *>(parent)->get_reward_at(index);
            float new_value = value + rtg;
            StatsManager::update_stats(parent, index, new_value, opt.game_count, opt.virtual_loss);
        }
    }
}

vec<BaseNode *> Path::get_all_nodes() const
{
    auto ret = vec<BaseNode *>();
    for (size_t i = 0; i < subpaths.size(); ++i)
    {
        const auto &subpath = subpaths[i];
        ret.push_back(tree_nodes[i]);
        for (auto node = subpath.mini_node_seq.begin(); node != subpath.mini_node_seq.end(); ++node)
            ret.push_back(static_cast<BaseNode *>(*node));
    }
    ret.push_back(tree_nodes.back());
    return ret;
}

vec<size_t> Path::get_all_chosen_indices() const
{
    auto ret = vec<size_t>();
    for (const auto &subpath : subpaths)
        for (const auto &chosen_char : subpath.chosen_seq)
            ret.push_back(chosen_char.first);
    return ret;
}

vec<abc_t> Path::get_all_chosen_actions() const
{
    auto ret = vec<abc_t>();
    for (const auto &subpath : subpaths)
        for (const auto &chosen_char : subpath.chosen_seq)
            ret.push_back(chosen_char.second);
    return ret;
}

void Path::merge(const Path &other)
{
    assert(other.subpaths.size() == 1);
    assert(tree_nodes.back() == other.tree_nodes.front());
    append(other.subpaths[0], other.tree_nodes[1]);
}

TreeNode *Path::get_last_node() const { return tree_nodes.back(); }

Path::Path(const Path &other)
{
    subpaths = other.subpaths;
    tree_nodes = other.tree_nodes;
    depth = other.depth;
}
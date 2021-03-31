#include "env.hpp"

Env::Env(const EnvOpt &env_opt, const ActionSpaceOpt &as_opt, const WordSpaceOpt &ws_opt) : opt(env_opt)
{
    assert(opt.start_ids.size() == opt.end_ids.size());

    // Set up the word space properly.
    word_space = new WordSpace(ws_opt, opt.end_ids);

    // Set up start and end states.
    size_t n = opt.start_ids.size();
    auto start_words = vec<Word *>();
    start_words.reserve(n);
    for (const auto &id_seq : opt.start_ids)
        start_words.push_back(word_space->get_word(id_seq));
    for (int order = 0; order < n; ++order)
    {
        word_space->set_edit_dist_at(start_words[order], order);
        word_space->set_edit_dist_at(word_space->end_words[order], order);
    }

    // start = new TreeNode(start_words, 0);
    // end = new TreeNode(word_space->end_words, node::END_DEPTH);
    start = NodeFactory::get_tree_node(start_words);
    end = NodeFactory::get_tree_node(word_space->end_words);

    // Set up the action space properly.
    action_space = new ActionSpace(word_space, as_opt);
    action_space->expand(start);

    // Set up lru cache.
    cache = LruCache();
}

TreeNode *Env::apply_action(TreeNode *node, const Subpath &subpath)
{
    auto *last = static_cast<TransitionNode *>(subpath.mini_node_seq[5]);
    int last_child_index = subpath.chosen_seq[6].first;
    // BaseNode *&child = last->children[last_child_index];
    BaseNode *child;
    if (!last->has_child(last_child_index))
    {
        child = action_space->apply_new_action(node, subpath);
        float reward;
        if (subpath.stopped)
            reward = -opt.step_penalty;
        {
            auto *tchild = static_cast<TreeNode *>(child);
            // reward = (node->dist - tchild->dist);
            float final_reward = tchild->is_done() ? opt.final_reward : -opt.step_penalty;
            float incremental_reward = (node->get_dist() - tchild->get_dist()) / start->get_dist();
            reward = final_reward + incremental_reward;
        }
        RewardManager::set_reward_at(last, last_child_index, reward);
    }
    else
        child = last->get_child(last_child_index);
    for (const auto node : subpath.mini_node_seq)
        cache.put(static_cast<BaseNode *>(node));
    cache.put(static_cast<BaseNode *>(child));
    return static_cast<TreeNode *>(child);
}

size_t Env::evict(size_t until_size)
{
    size_t size_before = cache.size();
    SPDLOG_TRACE("Before evicting #items: {}", size_before);
    // std::cerr << cache.persistent_size() << "\n";
    int num_to_evict = cache.size() - until_size - cache.persistent_size();
    while (num_to_evict-- > 0)
        cache.evict();
    // action_space->prune(base, true);
    SPDLOG_TRACE("After evicting #items: {}", cache.size());
    return size_before;
};

#include "env.hpp"

Env::Env(const EnvOpt &env_opt, const ActionSpaceOpt &as_opt, const WordSpaceOpt &ws_opt) : opt(env_opt)
{
    assert(opt.start_ids.size() == opt.end_ids.size());

    // Set up the word space properly.
    word_space = new WordSpace(opt.end_ids, ws_opt);

    // Set up start and end states.
    size_t n = opt.start_ids.size();
    auto start_words = vec<Word *>();
    start_words.reserve(n);
    for (const auto &id_seq : opt.start_ids)
        start_words.push_back(word_space->get_word(id_seq));
    for (int order = 0; order < n; ++order)
        word_space->set_edit_dist_at(start_words[order], order);

    start = new TreeNode(start_words, 0);
    end = new TreeNode(word_space->end_words, node::END_DEPTH);

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
    // Lock the last node since we are modifying its members.
    std::lock_guard<std::mutex> lock(last->mtx);
    BaseNode *&child = last->children[last_child_index];
    if (child == nullptr)
    {
        child = action_space->apply_new_action(node, subpath);
        float reward;
        if (subpath.stopped)
            reward = -opt.step_penalty;
        {
            auto *tchild = static_cast<TreeNode *>(child);
            // reward = (node->dist - tchild->dist);
            float final_reward = tchild->done ? opt.final_reward : -opt.step_penalty;
            float incremental_reward = (node->dist - tchild->dist) / start->dist;
            reward = final_reward + incremental_reward;
        }
        last->rewards[last_child_index] = reward;
    }
    for (const auto node : subpath.mini_node_seq)
        cache.put(node);
    cache.put(child);
    return static_cast<TreeNode *>(child);
}

void Env::evict(size_t until_size)
{
    while (cache.size() > until_size)
    {
        auto base = cache.evict();
        action_space->prune(base, true);
    }
};
#include "env.hpp"

Env::Env(const EnvOpt &env_opt, const WordSpaceOpt &ws_opt) : opt(env_opt)
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
        word_space->set_edit_dist(start_words[order], order);

    start = new TreeNode(start_words, 0);
    end = new TreeNode(word_space->end_words, node::END_DEPTH);
    action_space->expand(start);

    // Set up the action space properly.
    action_space = new ActionSpace(word_space);
}

TreeNode *Env::apply_action(TreeNode *node, const Subpath &subpath)
{
    MiniNode *last = subpath.second[4];
    int last_child_index = subpath.first[5].first;
    BaseNode *&child = last->children[last_child_index];
    if (child == nullptr)
    {
        child = action_space->apply_new_action(node, subpath);
        auto *tchild = static_cast<TreeNode *>(child);
        float final_reward = tchild->done ? opt.final_reward : -opt.step_penalty;
        float incremental_reward = (node->dist - tchild->dist) / start->dist;
        float reward = final_reward + incremental_reward;
        node->rewards[last_child_index] = reward;
    }
    return static_cast<TreeNode *>(child);
}
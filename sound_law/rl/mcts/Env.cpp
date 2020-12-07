#include "Env.hpp"
#include "TreeNode.hpp"

Env::Env(
    WordSpace *word_space,
    ActionSpace *action_space,
    const VocabIdSeq &start_ids,
    float final_reward,
    float step_penalty) : word_space(word_space),
                          action_space(action_space),
                          final_reward(final_reward),
                          step_penalty(step_penalty)
{
    // Obtaining the start here is safe since `word_space` has already taken care of `dist_mat` and `ins_cost`.
    std::vector<Word *> start_words = std::vector<Word *>();
    for (size_t order = 0; order < start_ids.size(); order++)
        start_words.push_back(word_space->get_word(start_ids.at(order), order, false));
    start = new TreeNode(start_words);
    end = new TreeNode(word_space->end_words);
}

TreeNode *Env::apply_action(TreeNode *node, action_t best_i, action_t action_id)
{
    Action action = action_space->get_action(action_id);
    return apply_action(node, action, best_i, action_id);
}

TreeNode *Env::apply_action(TreeNode *node, const Action &action, action_t best_i, action_t action_id)
{
    // Return nullptr if stop action is being applied.
    if (action.at(0) == NULL_abc)
        return nullptr;

    // Return cache if it exists. Obtain a read lock first.
    {
        boost::shared_lock_guard<boost::shared_mutex> lock(node->neighbor_mtx);
        if (node->neighbors.find(action_id) != node->neighbors.end())
            return node->neighbors.at(action_id);
    }

    // Obtain new list of words (by using cache whenever possbile).
    std::vector<Word *> new_words = std::vector<Word *>();
    for (size_t order = 0; order < node->words.size(); order++)
        new_words.push_back(word_space->apply_action(node->words.at(order), action, order));
    std::pair<action_t, action_t> prev_action = std::pair<action_t, action_t>(best_i, action_id);
    TreeNode *new_node = new TreeNode(new_words, prev_action, node);

    // Store it in neighbors.
    float final_reward = new_node->done ? this->final_reward : -this->step_penalty;
    float incremental_reward = (node->dist - new_node->dist) / start->dist;
    // Obtain the write lock before update. Make sure it hasn't been updated before -- otherwise release the memory.
    boost::lock_guard<boost::shared_mutex> lock(node->neighbor_mtx);
    if (node->neighbors.find(action_id) == node->neighbors.end())
    {
        node->neighbors[action_id] = new_node;
        node->rewards[action_id] = final_reward + incremental_reward;
        return new_node;
    }
    else
    {
        delete new_node;
        return node->neighbors.at(action_id);
    }
}
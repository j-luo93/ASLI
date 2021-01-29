#include "env.hpp"

Env::Env(ActionSpace *action_space,
         WordSpace *word_space,
         const VocabIdSeq &start_ids,
         const VocabIdSeq &end_ids,
         float final_reward,
         float step_penalty) : action_space(action_space),
                               word_space(word_space),
                               final_reward(final_reward),
                               step_penalty(step_penalty)
{
    // Set up the word space properly.
    auto end_words = vec<Word *>();
    word_space->get_words(nullptr, end_words, end_ids, false, end_ids.size());
    word_space->set_end_words(end_words);
    auto start_words = vec<Word *>();
    word_space->get_words(nullptr, start_words, start_ids);
    for (size_t order = 0; order < start_words.size(); order++)
    {
        auto word = start_words.at(order);

        word->dists.set(order, word_space->get_edit_dist(word, word_space->end_words.at(order)));
        end_words.at(order)->dists.set(order, 0.0);
    }
    start = new TreeNode(start_words, 0);
    end = new TreeNode(end_words, -1);
}

TreeNode *Env::apply_action(TreeNode *node, int best_i, uai_t action)
{
    std::lock_guard<std::mutex> lock(node->neighbor_mtx);
    auto &neighbors = node->neighbors;
    if (neighbors.find(action) != neighbors.end())
        return neighbors.at(action);

    auto prev_action = pair<int, uai_t>(best_i, action);
    TreeNode *new_node;
    float reward;

    // Special treatment for stop action.
    if (action == action::STOP)
    {
        new_node = new TreeNode(node->words, prev_action, node, true);
        // Still need to apply this for the last step.
        reward = -this->step_penalty;
        SPDLOG_DEBUG("Stop action applied.");
    }
    else
    {
        // Obtain new list of words.
        SPDLOG_DEBUG("Applying action {0}, best_i {1}.", action::str(action), best_i);
        auto new_words = vec<Word *>();
        int i = 0;
        for (const auto word : node->words)
        {
            // if (word->neighbors.find(action) == word->neighbors.end())
            // {
            //     assert(word->id_seq == action_space->apply_action(word->id_seq, action));
            //     new_words.push_back(word);
            // }
            // else
            usi_t site = action::get_site(action);
            bool pushed = false;
            if (word->neighbors.contains(site))
            {
                abc_t before_id = action::get_before_id(action);
                abc_t after_id = action::get_after_id(action);
                auto &edges = action_space->edges[before_id];
                for (size_t i = 0; i < edges.size(); i++)
                    if (edges[i] == after_id)
                    {
                        new_words.push_back(word->neighbors[site][i]);
                        pushed = true;
                        break;
                    }
            }
            if (!pushed)
                new_words.push_back(word);
            i++;
        }
        new_node = new TreeNode(new_words, prev_action, node, false);

        float final_reward = new_node->done ? this->final_reward : -this->step_penalty;
        float incremental_reward = (node->dist - new_node->dist) / start->dist;
        reward = final_reward + incremental_reward;
    }
    node->neighbors[action] = new_node;
    node->rewards[action] = reward;
    return new_node;
}
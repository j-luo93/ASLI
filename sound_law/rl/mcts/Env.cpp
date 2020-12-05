#include "Env.hpp"
#include "TreeNode.hpp"

Env::Env(
    WordSpace *word_space,
    ActionSpace *action_space,
    const VocabIdSeq &start_ids) : word_space(word_space),
                                   action_space(action_space)
{
    // Obtaining the start here is safe since `word_space` has already taken care of `dist_mat` and `ins_cost`.
    std::vector<Word *> start_words = std::vector<Word *>();
    for (size_t order = 0; order < start_ids.size(); order++)
        start_words.push_back(word_space->get_word(start_ids.at(order), order, false));
    start = new TreeNode(start_words);
    end = new TreeNode(word_space->end_words);
}

TreeNode *Env::apply_action(TreeNode *node, const Action &action)
{
    action_t action_id = action_space->get_action_id(action);
    // Return cache if it exists.
    if (node->neighbors.find(action_id) != node->neighbors.end())
        return node->neighbors.at(action_id);

    // Obtain new list of words (by using caching whenever possbile).
    std::vector<Word *> new_words = std::vector<Word *>();
    for (size_t order = 0; order < node->words.size(); order++)
        new_words.push_back(word_space->apply_action(node->words.at(order), action, order));
    TreeNode *new_node = new TreeNode(new_words);

    // Store it in neighbors.
    node->neighbors[action_id] = new_node;
    return new_node;
}
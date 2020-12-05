#include "Env.hpp"
#include "TreeNode.hpp"

Env::Env(
    WordSpace *word_space,
    ActionSpace *action_space,
    const VocabIdSeq &start_ids,
    const VocabIdSeq &end_ids) : word_space(word_space),
                                 action_space(action_space)
{
    std::vector<Word *> start_words = std::vector<Word *>();
    std::vector<Word *> end_words = std::vector<Word *>();
    for (const IdSeq &id_seq : start_ids)
        start_words.push_back(word_space->get_word(id_seq));
    for (const IdSeq &id_seq : end_ids)
        end_words.push_back(word_space->get_word(id_seq));

    start = new TreeNode(start_words);
    end = new TreeNode(end_words);
}

TreeNode *Env::apply_action(TreeNode *node, const Action &action)
{
    action_t action_id = action_space->get_action_id(action);
    // Return cache if it exists.
    if (node->neighbors.find(action_id) != node->neighbors.end())
        return node->neighbors.at(action_id);

    // Obtain new list of words (by using caching whenever possbile).
    std::vector<Word *> new_words = std::vector<Word *>();
    for (Word *word : node->words)
        new_words.push_back(word_space->apply_action(word, action));
    TreeNode *new_node = new TreeNode(new_words);

    // Store it in neighbors.
    node->neighbors[action_id] = new_node;
    return new_node;
}
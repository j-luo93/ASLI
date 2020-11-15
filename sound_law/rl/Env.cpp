#include <Env.h>
#include <algorithm>

Env::Env(TreeNode *init_node, TreeNode *end_node, ActionSpace *action_space, float final_reward, float step_penalty)
{
    this->init_node = init_node;
    this->end_node = end_node;
    this->action_space = action_space;
    this->final_reward = final_reward;
    this->step_penalty = step_penalty;
    this->starting_dist = init_node->dist_to_end;
};

Edge Env::step(TreeNode *node, long best_i, Action *action)
{
    assert(node->end_node != nullptr);
    if (node->has_acted(action->action_id))
    {
        return node->edges[action->action_id];
    }

    VocabIdSeq vocab_i = VocabIdSeq(node->size());
    for (long i = 0; i < node->size(); ++i)
    {
        vocab_i[i] = action->apply_to(node->vocab_i[i]);
    };
    vector<long> action_allowed = this->action_space->get_action_allowed(vocab_i);
    TreeNode *new_node = new TreeNode(vocab_i, node->end_node, best_i, action->action_id, node, action_allowed);
    float final_reward = new_node->done ? this->final_reward : -this->step_penalty;
    float incremental_reward = (node->dist_to_end - new_node->dist_to_end) / this->starting_dist;
    Edge edge = Edge(new_node, final_reward + incremental_reward);
    node->add_edge(action->action_id, edge);
    return edge;
};

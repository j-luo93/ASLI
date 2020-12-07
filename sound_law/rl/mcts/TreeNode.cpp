#include "TreeNode.hpp"

void TreeNode::common_init()
{
    dist = 0.0;
    for (auto word : words)
        dist += word->dist;

    done = true;
    for (auto word : words)
        if (!word->done)
        {
            done = false;
            break;
        }

    clear_stats();
}

TreeNode::TreeNode(const std::vector<Word *> &words) : words(words) { common_init(); }
TreeNode::TreeNode(
    const std::vector<Word *> &words,
    const std::pair<action_t, action_t> &prev_action,
    TreeNode *parent_node) : words(words),
                             parent_node(parent_node),
                             prev_action(prev_action) { common_init(); }

void TreeNode::debug()
{
    std::cerr << "Debug TreeNode: #action=" << action_allowed.size() << '\n';
    for (Word *word : words)
    {
        std::cerr << "---------\n";
        word->debug();
    }
}

bool TreeNode::is_leaf() { return prior.size() == 0; }

std::vector<float> TreeNode::get_scores(float puct_c)
{
    float sqrt_ns = sqrt((float)this->visit_count);
    std::vector<float> scores = std::vector<float>(this->prior.size());
    for (size_t i = 0; i < this->prior.size(); i++)
    {
        float nsa = (float)this->action_count[i];
        float q = this->total_value[i] / (nsa + 1e-8);
        float p = this->prior[i];
        float u = puct_c * p * sqrt_ns / (1 + nsa);
        scores[i] = q + u;
    }
    return scores;
}

action_t TreeNode::get_best_i(float puct_c)
{
    action_t best_i = NULL_action;
    float best_v;
    std::vector<float> scores = this->get_scores(puct_c);
    for (size_t i = 0; i < this->prior.size(); ++i)
    {
        if ((best_i == NULL_action) or (scores[i] > best_v))
        {
            best_v = scores[i];
            best_i = i;
        };
    }
    assert(best_i != NULL_action);
    return best_i;
}

void TreeNode::expand(const std::vector<float> &prior)
{
    clear_stats();
    this->prior = prior;
    size_t num_actions = action_allowed.size();
    assert(num_actions == prior.size());
    action_count = std::vector<visit_t>(num_actions, 0);
    total_value = std::vector<float>(num_actions, 0.0);
}

void TreeNode::clear_stats(bool recursive)
{
    action_count.clear();
    total_value.clear();
    visit_count = 0;
    max_value = -9999.9;
    max_index = -1;
    max_action_id = NULL_action;
    played = false;

    if (recursive)
        for (const auto &item : neighbors)
            item.second->clear_stats(recursive);
}

action_t TreeNode::select(float puct_c, int game_count, float virtual_loss)
{
    select_mtx.lock();
    action_t best_i = get_best_i(puct_c);
    virtual_backup(best_i, game_count, virtual_loss);
    select_mtx.unlock();
    return best_i;
}

void TreeNode::virtual_backup(action_t best_i, int game_count, float virtual_loss)
{
    action_count[best_i] += game_count;
    total_value[best_i] -= game_count * virtual_loss;
    visit_count += game_count;
}

void TreeNode::backup(float value, float mixing, int game_count, float virtual_loss)
{
    TreeNode *parent_node = this->parent_node;
    TreeNode *node = this;
    float rtg = 0.0;
    while ((parent_node != nullptr) and (!parent_node->played))
    {
        std::pair<action_t, action_t> &pa = node->prev_action;
        action_t best_i = pa.first;
        action_t action_id = pa.second;
        float reward = parent_node->rewards.at(action_id);
        rtg += reward;
        // float mixed_value = (1 - mixing) * value + mixing * reward;
        parent_node->action_count[best_i] -= game_count - 1;
        assert(parent_node->action_count[best_i] >= 0);
        // parent_node->total_value[action_id] += game_count * virtual_loss + mixed_value;
        // Update max value of the parent.
        float new_value = value + rtg;
        if (new_value > parent_node->max_value)
        {
            parent_node->max_value = new_value;
            parent_node->max_index = best_i;
            parent_node->max_action_id = action_id;
        }
        parent_node->total_value[best_i] += game_count * virtual_loss + new_value;
        parent_node->visit_count -= game_count - 1;
        node = parent_node;
        parent_node = node->parent_node;
    }
}

void TreeNode::play()
{
    assert(!played);
    played = true;
}

std::list<std::pair<action_t, float>> TreeNode::get_path()
{
    TreeNode *node = this;
    std::list<std::pair<action_t, float>> path = std::list<std::pair<action_t, float>>();
    while (node->parent_node != nullptr)
    {
        action_t action_id = node->prev_action.second;
        float reward = node->parent_node->rewards.at(action_id);
        std::pair<action_t, float> path_edge = std::pair<action_t, float>{action_id, reward};
        path.push_front(path_edge);
        node = node->parent_node;
    }
    return path;
}

void TreeNode::add_noise(const std::vector<float> &noise, float noise_ratio)
{
    assert(!prior.empty());
    assert(noise.size() == prior.size());
    for (size_t i = 0; i < prior.size(); ++i)
        prior[i] = prior[i] * (1.0 - noise_ratio) + noise[i] * noise_ratio;
}

size_t TreeNode::size() { return words.size(); }
size_t DetachedTreeNode::size() { return vocab_i.size(); }
IdSeq TreeNode::get_id_seq(int order) { return words.at(order)->id_seq; }
IdSeq DetachedTreeNode::get_id_seq(int order) { return vocab_i.at(order); }
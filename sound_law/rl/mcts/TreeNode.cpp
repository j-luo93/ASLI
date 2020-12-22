#include "TreeNode.hpp"
#include "Word.hpp"

boost::mutex TreeNode::cls_mtx;
unsigned long TreeNode::cls_cnt = 0;

void TreeNode::common_init()
{
    dist = 0.0;
    for (int order = 0; order < words.size(); order++)
    {
        auto word = words.at(order);
        dist += word->dists.at(order);
    }

    done = true;
    for (auto word : words)
        if (!word->done)
        {
            done = false;
            break;
        }

    clear_stats();

    {
        boost::lock_guard<boost::mutex> lock(TreeNode::cls_mtx);
        idx = TreeNode::cls_cnt++;
    }
}

TreeNode::TreeNode(const std::vector<Word *> &words) : words(words) { common_init(); }
TreeNode::TreeNode(
    const std::vector<Word *> &words,
    const std::pair<int, uai_t> &prev_action,
    TreeNode *parent_node,
    bool stopped) : words(words),
                    stopped(stopped),
                    parent_node(parent_node),
                    prev_action(prev_action) { common_init(); }

void TreeNode::debug(bool show_words)
{
    std::cerr << "Debug TreeNode: #action=" << action_allowed.size() << '\n';
    std::cerr << "stopped: " << stopped << " done: " << done << '\n';
    if (show_words)
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

int TreeNode::get_best_i(float puct_c)
{
    int best_i = -1;
    float best_v;
    std::vector<float> scores = this->get_scores(puct_c);
    for (size_t i = 0; i < this->prior.size(); ++i)
    {
        if ((best_i == -1) or (scores[i] > best_v))
        {
            best_v = scores[i];
            best_i = i;
        };
    }
    assert(best_i != -1);
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
    prior.clear();
    visit_count = 0;
    max_value = -9999.9;
    max_index = -1;
    max_action_id = NULL_ACTION;
    played = false;

    if (recursive)
        for (const auto &item : neighbors)
            item.second->clear_stats(recursive);
}

int TreeNode::select(float puct_c, int game_count, float virtual_loss)
{
    boost::lock_guard<boost::mutex> lock(exclusive_mtx);
    int best_i = get_best_i(puct_c);
    virtual_backup(best_i, game_count, virtual_loss);
    return best_i;
}

void TreeNode::virtual_backup(int best_i, int game_count, float virtual_loss)
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
        std::pair<int, uai_t> &pa = node->prev_action;
        int best_i = pa.first;
        uai_t action_id = pa.second;
        float reward = parent_node->rewards.at(action_id);
        rtg += reward;
        // float mixed_value = (1 - mixing) * value + mixing * reward;
        parent_node->action_count[best_i] -= game_count - 1;
        // assert(parent_node->action_count[best_i] >= 1);
        if (parent_node->action_count[best_i] < 1)
        {
            std::cerr << best_i << '\n';
            std::cerr << action_id << '\n';
            std::cerr << parent_node->action_count[best_i] << '\n';
            assert(false);
        }
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

std::list<std::pair<uai_t, float>> TreeNode::get_path()
{
    TreeNode *node = this;
    auto path = std::list<std::pair<uai_t, float>>();
    while (node->parent_node != nullptr)
    {
        uai_t action_id = node->prev_action.second;
        float reward = node->parent_node->rewards.at(action_id);
        auto path_edge = std::pair<uai_t, float>{action_id, reward};
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

size_t TreeNode::get_num_descendants()
{
    size_t ret = 1;
    for (const auto &item : neighbors)
    {
        ret += item.second->get_num_descendants();
    }
    return ret;
}

size_t TreeNode::clear_cache(float ratio_threshold)
{
    size_t num_desc = 1;
    auto to_remove = std::vector<uai_t>();
    for (const auto &item : neighbors)
    {
        auto neighbor = item.second;
        size_t inc = neighbor->clear_cache(ratio_threshold);
        float ratio = static_cast<float>(neighbor->visit_count) / static_cast<float>(inc);
        if (ratio < ratio_threshold)
            to_remove.push_back(item.first);
        num_desc += inc;
    }
    for (const auto action_id : to_remove)
    {
        delete neighbors.at(action_id);
        neighbors.erase(action_id);
    }
    return num_desc;
}

size_t TreeNode::size() { return words.size(); }
size_t DetachedTreeNode::size() { return vocab_i.size(); }
IdSeq TreeNode::get_id_seq(int order) { return words.at(order)->id_seq; }
IdSeq DetachedTreeNode::get_id_seq(int order) { return vocab_i.at(order); }

DetachedTreeNode::DetachedTreeNode(TreeNode *node) : action_allowed(node->action_allowed)
{
    vocab_i = VocabIdSeq(node->size());
    for (size_t order = 0; order < node->size(); order++)
        vocab_i[order] = node->get_id_seq(order);
}
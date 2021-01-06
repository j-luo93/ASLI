#include "tree_node.hpp"

void TreeNode::common_init()
{
    dist = 0.0;
    done = true;
    for (size_t order = 0; order < words.size(); order++)
    {
        auto word = words.at(order);
        auto w_dist = word->dists.get(order);
        dist += w_dist;
        done = done && (w_dist == 0.0);
    }
    clear_stats();
}

TreeNode::TreeNode(const vec<Word *> &words, int depth) : words(words), depth(depth) { common_init(); };
TreeNode::TreeNode(
    const vec<Word *> &words,
    const pair<int, uai_t> &prev_action,
    TreeNode *parent_node,
    bool stopped) : words(words),
                    prev_action(prev_action),
                    parent_node(parent_node),
                    stopped(stopped)
{
    common_init();
    assert(parent_node->depth >= 0);
    depth = parent_node->depth + 1;
}

std::string TreeNode::str()
{
    std::string out = "depth " + std::to_string(depth) + " dist " + std::to_string(dist) + "\n";
    out += ">>>>>>>>>>\n";
    for (size_t i = 0; i < words.size(); i++)
        out += "word #" + std::to_string(i) + ": " + words.at(i)->str() + '\n';
    out += "<<<<<<<<<<";
    return out;
}

bool TreeNode::is_leaf() { return prior.size() == 0; }

int TreeNode::select(float puct_c, int game_count, float virtual_loss)
{
    std::lock_guard<std::mutex> lock(score_mtx);
    int best_i = get_best_i(puct_c);
    action_count[best_i] += game_count;
    total_value[best_i] -= game_count * virtual_loss;
    visit_count += game_count;
    return best_i;
}

vec<float> TreeNode::get_scores(float puct_c)
{
    float sqrt_ns = sqrt(static_cast<float>(visit_count));
    auto scores = vec<float>(prior.size());
    for (size_t i = 0; i < prior.size(); i++)
    {
        float nsa = static_cast<float>(action_count[i]);
        float q = total_value[i] / (nsa + 1e-8);
        float p = prior[i];
        float u = puct_c * p * sqrt_ns / (1 + nsa);
        scores[i] = q + u;
    }
    return scores;
}

int TreeNode::get_best_i(float puct_c)
{
    auto scores = get_scores(puct_c);
    auto it = std::max_element(scores.begin(), scores.end());
    return std::distance(scores.begin(), it);
}

void TreeNode::expand(const vec<float> &prior)
{
    clear_stats();
    this->prior = prior;
    size_t num_actions = action_allowed.size();
    assert(num_actions == prior.size());
    action_count = std::vector<visit_t>(num_actions, 0);
    total_value = std::vector<float>(num_actions, 0.0);
}

void TreeNode::backup(float value, int game_count, float virtual_loss)
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
        parent_node->action_count[best_i] -= game_count - 1;
        if (parent_node->action_count[best_i] < 1)
        {
            std::cerr << best_i << '\n';
            std::cerr << action_id << '\n';
            std::cerr << parent_node->action_count[best_i] << '\n';
            assert(false);
        }
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

size_t TreeNode::get_num_descendants()
{
    size_t ret = 1;
    for (const auto &item : neighbors)
    {
        ret += item.second->get_num_descendants();
    }
    return ret;
}

void TreeNode::add_noise(const std::vector<float> &noise, float noise_ratio)
{
    assert(!prior.empty());
    assert(noise.size() == prior.size());
    for (size_t i = 0; i < prior.size(); ++i)
        prior[i] = prior[i] * (1.0 - noise_ratio) + noise[i] * noise_ratio;
}

IdSeq TreeNode::get_id_seq(int order) { return words.at(order)->id_seq; }
IdSeq DetachedTreeNode::get_id_seq(int order) { return vocab_i.at(order); }

size_t TreeNode::size() { return words.size(); }
size_t DetachedTreeNode::size() { return vocab_i.size(); }

DetachedTreeNode::DetachedTreeNode(TreeNode *node) : action_allowed(node->action_allowed)
{
    size_t size = node->words.size();
    vocab_i = VocabIdSeq(size);
    for (size_t order = 0; order < size; order++)
        vocab_i[order] = node->get_id_seq(order);
}
#include "node.hpp"

BaseNode::BaseNode(BaseNode *const parent, const ChosenChar &chosen_char, bool stopped) : parent(parent), chosen_char(chosen_char), stopped(stopped) {}

MiniNode::MiniNode(TreeNode *base, BaseNode *const parent, const ChosenChar &chosen_char, ActionPhase ap, bool stopped) : base(base), ap(ap), BaseNode(parent, chosen_char, stopped) {}

TransitionNode::TransitionNode(TreeNode *base,
                               MiniNode *parent,
                               const ChosenChar &chosen_char,
                               bool stopped) : MiniNode(base, static_cast<BaseNode *>(parent), chosen_char, ActionPhase::POST, stopped) {}

void TreeNode::common_init(const vec<Word *> &words)
{
    for (int order = 0; order < words.size(); ++order)
        dist += words[order]->get_edit_dist_at(order);

    if (dist == 0.0)
        done = true;
}

TreeNode::TreeNode(const vec<Word *> &words,
                   int depth) : words(words),
                                depth(depth),
                                BaseNode(nullptr, ChosenChar(-1, 0), false) { common_init(words); }

TreeNode::TreeNode(const vec<Word *> &words,
                   int depth,
                   BaseNode *const parent,
                   const ChosenChar &chosen_char,
                   bool stopped) : words(words),
                                   depth(depth),
                                   BaseNode(parent, chosen_char, stopped) { common_init(words); }

bool BaseNode::is_expanded() { return (permissible_chars.size() > 0); }
bool BaseNode::is_evaluated() { return (priors.size() > 0); }

ChosenChar BaseNode::get_best_subaction(float puct_c, int game_count, float virtual_loss)
{
    std::lock_guard<std::mutex> lock(mtx);
    assert(is_expanded() && is_evaluated());
    auto scores = get_scores(puct_c);
    auto it = std::max_element(scores.begin(), scores.end());
    int index = std::distance(scores.begin(), it);
    auto ret = ChosenChar(index, permissible_chars[index]);
    action_counts[index] += game_count;
    total_values[index] -= game_count * virtual_loss;
    visit_count += game_count;
    SPDLOG_DEBUG("BaseNode: getting best subaction ({0}, {1})", ret.first, ret.second);
    return ret;
}

vec<float> BaseNode::get_scores(float puct_c)
{
    float sqrt_ns = sqrt(static_cast<float>(visit_count));
    auto scores = vec<float>(priors.size());
    for (size_t i = 0; i < priors.size(); ++i)
    {
        float nsa = static_cast<float>(action_counts[i]);
        float q = total_values[i] / (nsa + 1e-8);
        float p = priors[i];
        float u = puct_c * p * sqrt_ns / (1 + nsa);
        scores[i] = q + u;
    }
    return scores;
}

bool TreeNode::is_leaf() { return priors.size() == 0; }

TreeNode *TreeNode::play()
{
    // std::cerr << "============================\n";
    MiniNode *mini_node = static_cast<MiniNode *>(BaseNode::play());
    for (int i = 0; i < 5; ++i)
        mini_node = static_cast<MiniNode *>(mini_node->play());
    return static_cast<TreeNode *>(mini_node->play());
}

BaseNode *BaseNode::play()
{
    // auto scores = get_scores(5.0);
    // std::cerr << "-------------------------\nPLAY:\n";
    // for (size_t i = 0; i < scores.size(); ++i)
    //     std::cerr << permissible_chars[i] << ":" << scores[i] << " ";
    // std::cerr << "\n";
    // std::cerr << "max index: " << max_index << " char: " << permissible_chars[max_index] << " max_value: " << max_value << "\n";
    assert(max_index != -1);
    assert(!played);
    played = true;
    return children[max_index];
}

void BaseNode::backup(float value, int game_count, float virtual_loss)
{
    BaseNode *parent = this->parent;
    BaseNode *node = this;
    float rtg = 0.0;
    assert(parent != nullptr);
    while ((parent != nullptr) and (!parent->played))
    {
        auto &chosen = node->chosen_char;
        int index = chosen.first;
        abc_t best_char = chosen.second;
        // auto tparent = dynamic_cast<TransitionNode *>(parent);
        // if (tparent != nullptr)
        //     rtg += tparent->rewards[index];
        if (parent->is_transitional())
            rtg += static_cast<TransitionNode *>(parent)->rewards[index];
        parent->action_counts[index] -= game_count - 1;
        if (parent->action_counts[index] < 1)
        {
            std::cerr << index << '\n';
            std::cerr << best_char << '\n';
            std::cerr << parent->action_counts[index] << '\n';
            assert(false);
        }
        // Update max value of the parent.
        float new_value = value + rtg;
        if (new_value > parent->max_value)
        {
            parent->max_value = new_value;
            parent->max_index = index;
        }
        parent->total_values[index] += game_count * virtual_loss + new_value;
        parent->visit_count -= game_count - 1;
        node = parent;
        parent = node->parent;
    }
}

IdSeq TreeNode::get_id_seq(int order) { return words[order]->id_seq; }

size_t TreeNode::size() { return words.size(); }

size_t BaseNode::get_num_actions()
{
    assert(permissible_chars.size() > 0);
    return permissible_chars.size();
}

void BaseNode::add_noise(const vec<float> &noise, float noise_ratio)
{
    assert(noise.size() == get_num_actions());
    for (size_t i = 0; i < priors.size(); ++i)
        priors[i] = priors[i] * (1.0 - noise_ratio) + noise[i] * noise_ratio;
}

bool MiniNode::is_transitional() { return false; }
bool TransitionNode::is_transitional() { return true; }
bool TreeNode::is_transitional() { return false; }
bool MiniNode::is_tree_node() { return false; }
bool TreeNode::is_tree_node() { return true; }
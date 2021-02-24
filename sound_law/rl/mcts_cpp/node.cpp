#include "node.hpp"

BaseNode::BaseNode(BaseNode *const parent, const ChosenChar &chosen_char, bool stopped, bool persistent) : stopped(stopped), persistent(persistent) {}

MiniNode::MiniNode(TreeNode *base, BaseNode *const parent, const ChosenChar &chosen_char, ActionPhase ap, bool stopped) : base(base), ap(ap), BaseNode(parent, chosen_char, stopped, false) {}

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
                                BaseNode(nullptr, ChosenChar(-1, 0), false, true) { common_init(words); }

TreeNode::TreeNode(const vec<Word *> &words,
                   int depth,
                   BaseNode *const parent,
                   const ChosenChar &chosen_char,
                   bool stopped) : words(words),
                                   depth(depth),
                                   BaseNode(parent, chosen_char, stopped, false) { common_init(words); }

bool BaseNode::is_expanded() { return (permissible_chars.size() > 0); }
bool BaseNode::is_evaluated() { return (priors.size() > 0); }

ChosenChar BaseNode::get_best_subaction(float puct_c, int game_count, float virtual_loss, float heur_c)
{
    std::lock_guard<std::mutex> lock(mtx);
    assert(is_expanded() && is_evaluated());
    auto scores = get_scores(puct_c, heur_c);
    auto it = std::max_element(scores.begin(), scores.end());
    int index = std::distance(scores.begin(), it);
    auto ret = ChosenChar(index, permissible_chars[index]);
    action_counts[index] += game_count;
    total_values[index] -= game_count * virtual_loss;
    visit_count += game_count;
    SPDLOG_DEBUG("BaseNode: getting best subaction ({0}, {1})", ret.first, ret.second);
    return ret;
}

inline float randf(float high)
{
    return high * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

vec<float> BaseNode::get_scores(float puct_c, float heur_c)
{
    float sqrt_ns = sqrt(static_cast<float>(visit_count)); // + 1;
    auto scores = vec<float>(priors.size());
    for (size_t i = 0; i < priors.size(); ++i)
    {
        float nsa = static_cast<float>(action_counts[i]);
        float q = total_values[i] / (nsa + 1e-8);
        float mv = max_values[i];
        float p = priors[i];
        float u = puct_c * p * sqrt_ns / (1 + nsa);
        // float h = heur_c * (static_cast<float>(affected[i].size())) / (1 + nsa);
        float h = heur_c * sqrt(static_cast<float>(affected[i].size())) / (1 + nsa);
        // float h = (mv > -99.9) ? heur_c * mv / (1 + nsa) : 0.0;
        // scores[i] = q + u + h;
        // scores[i] = q + u + randf(0.001);
        // scores[i] = pruned[i] ? -9999.9 : (q + u);
        // scores[i] = pruned[i] ? -9999.9 : (q + u + h + randf(0.01));
        scores[i] = pruned[i] ? -9999.9 : (q + u + h);
        // scores[i] = q + u; //+ h + randf(0.01);
    }
    return scores;
}

void BaseNode::prune()
{
    // std::cerr << "complete: " << num_unpruned_actions << "\n";
    SPDLOG_TRACE("Prune this node with #actions {}", num_unpruned_actions);
    num_unpruned_actions = 0;
    std::fill(pruned.begin(), pruned.end(), true);
    for (size_t i = 0; i < parents.size(); ++i)
        parents[i]->prune(parent_indices[i]);
    // if (parent != nullptr)
    //     parent->prune(chosen_char.first);
}

void BaseNode::prune(int index)
{
    // assert(!pruned[index]);
    SPDLOG_TRACE("Prune this node with #actions {0} at index {1}", num_unpruned_actions, index);
    if (!pruned[index])
    {
        pruned[index] = true;
        // std::cerr << "old: " << num_unpruned_actions << "\n";
        --num_unpruned_actions;
        // std::cerr << "new: " << num_unpruned_actions << "\n";
    }
    if (is_pruned())
        prune();
    // if (is_pruned() && (parent != nullptr))
    //     parent->prune(chosen_char.first);
}

bool BaseNode::is_pruned() { return num_unpruned_actions == 0; }

bool TreeNode::is_leaf() { return priors.size() == 0; }

pair<TreeNode *, Subpath> TreeNode::play()
{
    // std::cerr << "============================\n";
    BaseNode *node = this;
    auto subpath = Subpath();
    for (int i = 0; i < 7; ++i)
    {
        auto mini_ret = node->play_mini();
        if (i < 6)
            subpath.mini_node_seq[i] = static_cast<MiniNode *>(mini_ret.first);
        subpath.chosen_seq[i] = mini_ret.second;
        node = mini_ret.first;
    }
    return std::make_pair(static_cast<TreeNode *>(node), subpath);
}

pair<BaseNode *, ChosenChar> BaseNode::play_mini()
{
    // int index = 0;
    // for (size_t i = 1; i < permissible_chars.size(); ++i)
    //     if (action_counts[i] > action_counts[index])
    //         index = i;
    // assert(!played);
    // played = true;
    // return children[index];

    // std::cerr << "-------------------------\nPLAY:\n";
    // for (size_t i = 0; i < permissible_chars.size(); ++i)
    // {
    //     std::cerr << permissible_chars[i] << ":";
    //     std::cerr << action_counts[i] << ":";
    //     std::cerr << (pruned[i] ? "p" : "u") << " ";
    // }
    // std::cerr << "\n";
    // std::cerr << "max index: " << max_index << " char: " << permissible_chars[max_index] << " max_value: " << max_value << "\n";

    assert(max_index != -1);
    assert(!played);
    played = true;
    return std::make_pair(children[max_index], ChosenChar{max_index, permissible_chars[max_index]});

    // auto probs = vec<float>();
    // probs.reserve(action_counts.size());
    // float sum = 0.0;
    // for (size_t i = 0; i < max_values.size(); ++i)
    // {
    //     auto mv = max_values[i];
    //     if (mv > -99.9)
    //         if (pruned[i])
    //             probs.push_back(1e-8);
    //         else
    //             probs.push_back(exp(mv * 50.0));
    //     else
    //         probs.push_back(0.0);
    //     // probs.push_back((mv > -99.9) ? exp(mv * 50.0) : 0.0);
    //     // std::cerr << i << " " << probs.back() << " " << mv << " " << pruned[i] << "\n";
    //     sum += probs.back();
    // }

    // // for (const auto ac : action_counts)
    // // {
    // //     probs.push_back(pow(static_cast<float>(ac), 1.0));
    // //     sum += probs.back();
    // // }
    // for (auto &prob : probs)
    //     prob /= sum;

    // // auto scores = get_scores(1.0, 1.0);
    // // std::cerr << "-------------------------\nPLAY:\n";
    // // if (parent != nullptr)
    // //     std::cerr << "#unpruned: " << num_unpruned_actions << " "
    // //               << "#affected: " << parent->affected[chosen_char.first].size() << "\n";
    // // for (size_t i = 0; i < permissible_chars.size(); ++i)
    // // {
    // //     std::cerr << permissible_chars[i] << ":";
    // //     std::cerr << total_values[i] << "/" << action_counts[i] << "=";
    // //     std::cerr << total_values[i] / (1e-8 + action_counts[i]) << ":";
    // //     // std::cerr << max_values[i] << " ";
    // //     std::cerr << max_values[i] << ":";
    // //     std::cerr << ((children[i] == nullptr) ? -1 : children[i]->num_unpruned_actions) << ":";
    // //     std::cerr << probs[i] << " ";
    // // }
    // // std::cerr << "\n";
    // // std::cerr << "max index: " << max_index << " char: " << permissible_chars[max_index] << " max_value: " << max_value << "\n";

    // float r = randf(1.0);
    // float low = 0.0;
    // float high = 0.0;
    // size_t index = 0;
    // for (size_t i = 0; i < probs.size(); ++i)
    // {
    //     high += probs[i];
    //     if ((r >= low) && (r < high))
    //     {
    //         index = i;
    //         break;
    //     }
    //     low = high;
    // }
    // assert(!played);
    // // assert(!pruned[index]);
    // assert(children[index] != nullptr);
    // // assert(children[index]->num_unpruned_actions > 0);
    // played = true;
    // return children[index];
}

IdSeq TreeNode::get_id_seq(int order) { return words[order]->id_seq; }

size_t TreeNode::size() { return words.size(); }

size_t BaseNode::get_num_actions()
{
    assert(permissible_chars.size() > 0);
    return permissible_chars.size();
}

bool MiniNode::is_transitional() { return false; }
bool TransitionNode::is_transitional() { return true; }
bool TreeNode::is_transitional() { return false; }
bool MiniNode::is_tree_node() { return false; }
bool TreeNode::is_tree_node() { return true; }

// size_t BaseNode::get_num_descendants()
// {
//     size_t ret = 1;
//     for (const auto child : children)
//         if (child != nullptr)
//             ret += child->get_num_descendants();
//     return ret;
// }

Trie<Word *, TreeNode *> TreeNode::t_table = Trie<Word *, TreeNode *>(nullptr);

TreeNode *TreeNode::get_tree_node(const vec<Word *> &words, int depth)
{
    auto new_node = new TreeNode(words, depth);
    auto ret = new_node;
    if (TreeNode::t_table.get(words, ret))
        delete new_node;
    return ret;
}

TreeNode *TreeNode::get_tree_node(const vec<Word *> &words, int depth, BaseNode *const parent, const ChosenChar &chosen_char, bool stopped)
{
    auto new_node = new TreeNode(words, depth, parent, chosen_char, stopped);
    auto ret = new_node;
    if (TreeNode::t_table.get(words, ret))
        delete new_node;
    return ret;
}

bool BaseNode::has_child(size_t index) const
{
    assert(children.size() > index);
    return (children[index] != nullptr);
}

// Returns the child (including nullptr) at the index.
BaseNode *BaseNode::get_child(size_t index) const
{
    assert(children.size() > index);
    return children[index];
}

int BaseNode::get_in_degree() const { return in_degree; }

void BaseNode::disconnect_from_parents()
{
    for (size_t i = 0; i < parents.size(); ++i)
    {
        const auto parent = parents[i];
        const auto index = parent_indices[i];
        parent->children[index] = nullptr;
    }
    parents.clear();
    parent_indices.clear();
    in_degree = 0;
}

void BaseNode::disconnect_from_children()
{
    for (size_t i = 0; i < children.size(); ++i)
    {
        const auto child = children[i];
        if (child != nullptr)
        {
            --child->in_degree;
            auto it = std::find(child->parents.begin(), child->parents.end(), this);
            assert(it != child->parents.end());
            auto index = std::distance(child->parents.begin(), it);
            child->parents.erase(it);
            child->parent_indices.erase(child->parent_indices.begin() + index);
            children[i] = nullptr;
        }
    }
}

// BaseNode::~BaseNode()
// {
//     disconnect_from_parents();
//     disconnect_from_children();
// }

// TreeNode::~TreeNode()
// {
//     // Only remove it from the table if it is not stopped.
//     if (!stopped)
//         TreeNode::t_table.remove(words);
// }
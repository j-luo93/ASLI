#include "node.hpp"

BaseNode::BaseNode(bool stopped, bool persistent) : stopped(stopped), persistent(persistent) {}

MiniNode::MiniNode(const TreeNode *base, ActionPhase ap, bool stopped) : base(base), ap(ap), BaseNode(stopped, false) {}

TransitionNode::TransitionNode(const TreeNode *base,
                               bool stopped) : MiniNode(base, ActionPhase::POST, stopped) {}

void TreeNode::common_init(const vec<Word *> &words)
{
    for (int order = 0; order < words.size(); ++order)
        dist += words[order]->get_edit_dist_at(order);

    if (dist == 0.0)
        done = true;
}

TreeNode::TreeNode(const vec<Word *> &words) : words(words),
                                               BaseNode(false, true) { common_init(words); }

TreeNode::TreeNode(const vec<Word *> &words,
                   bool stopped) : words(words),
                                   BaseNode(stopped, false) { common_init(words); }

bool BaseNode::is_expanded() const { return (permissible_chars.size() > 0); }
bool BaseNode::is_evaluated() const { return (priors.size() > 0); }

ChosenChar BaseNode::get_best_action(const SelectionOpt &sel_opt) const
{
    assert(is_expanded() && is_evaluated());
    int index;
    if (sel_opt.random_select)
    {
        index = rand() % permissible_chars.size();
    }
    else if (sel_opt.policy_only)
    {
        index = 0;
        for (size_t i = 1; i < priors.size(); ++i)
            if (priors[i] > priors[index])
                index = i;
    }
    else
    {
        auto scores = get_scores(sel_opt);
        auto it = std::max_element(scores.begin(), scores.end());
        index = std::distance(scores.begin(), it);
    }
    auto ret = ChosenChar(index, permissible_chars[index]);
    SPDLOG_DEBUG("BaseNode: getting best subaction ({0}, {1})", ret.first, ret.second);
    return ret;
}

inline float randf(float high)
{
    return high * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

vec<float> BaseNode::get_scores(const SelectionOpt &sel_opt) const
{
    assert(!stopped || !is_tree_node());
    float sqrt_ns = sqrt(static_cast<float>(visit_count)); // + 1;
    auto scores = vec<float>(priors.size());
    assert(priors.size() == pruned.size());
    // std::cerr << "=======================================\n";
    for (size_t i = 0; i < priors.size(); ++i)
    {
        float nsa = static_cast<float>(action_counts[i]);
        float q;
        if (sel_opt.use_max_value)
            q = nsa > 0 ? max_values[i] : 0.0;
        else
            q = total_values[i] / (nsa + 1e-8);
        float p = priors[i];
        float u = sel_opt.puct_c * p * sqrt_ns / (1 + nsa);
        // float h = heur_c * (static_cast<float>(affected[i].size())) / (1 + nsa);
        // float h = heur_c * sqrt(static_cast<float>(affected[i].size())) / (1 + nsa);
        float h;
        if (sel_opt.use_num_misaligned)
            h = sel_opt.heur_c > 0.0 ? sel_opt.heur_c * static_cast<float>(affected[i].get_num_misaligned()) / (1 + nsa) : 0.0;
        else
            h = sel_opt.heur_c > 0.0 ? sel_opt.heur_c * affected[i].get_misalignment_score() / (1 + nsa) : 0.0;
        // std::cerr << permissible_chars[i] << ":" << h << " ";

        // std::cerr << "--------------\n";
        // std::cerr << affected[i].num_misaligned() << " ";
        // for (size_t j = 0; j < affected[i].size(); ++j)
        //     std::cerr << affected[i].positions[j] << " ";
        // std::cerr << "\n";
        // float h = (mv > -99.9) ? heur_c * mv / (1 + nsa) : 0.0;
        // scores[i] = q + u + h;
        // scores[i] = q + u + randf(0.001);
        // scores[i] = pruned[i] ? -9999.9 : (q + u);
        // scores[i] = pruned[i] ? -9999.9 : (q + u + h + randf(0.01));
        float noise = sel_opt.add_noise ? randf(1e-8) : 0.0;
        scores[i] = pruned[i] ? -9999.9 : (q + u + h + noise);
        // scores[i] = pruned[i] ? -9999.9 : (mv + u + h + noise);
        // scores[i] = q + u; //+ h + randf(0.01);
    }
    // std::cerr << "\n";
    return scores;
}

void BaseNode::prune()
{
    SPDLOG_TRACE("Prune this node with #actions {}", num_unpruned_actions);
    num_unpruned_actions = 0;
    std::fill(pruned.begin(), pruned.end(), true);
    for (size_t i = 0; i < parents.size(); ++i)
        parents[i]->prune(parent_indices[i]);
}

void BaseNode::prune(size_t index)
{
    SPDLOG_TRACE("Prune this node with #actions {0} at index {1}", num_unpruned_actions, index);
    if (!pruned[index])
    {
        pruned[index] = true;
        --num_unpruned_actions;
    }
    if (is_pruned())
        prune();
}

bool BaseNode::is_pruned() const { return num_unpruned_actions == 0; }

bool TreeNode::is_leaf() const { return priors.size() == 0; }

pair<TreeNode *, Subpath> TreeNode::play(PlayStrategy ps, float exponent) const
{
    SPDLOG_TRACE("Playing one step.");
    auto subpath = Subpath();
    auto mini_ret = play_mini(ps, exponent);
    BaseNode *node = mini_ret.first;
    subpath.mini_node_seq[0] = static_cast<MiniNode *>(node);
    subpath.chosen_seq[0] = mini_ret.second;
    for (int i = 1; i < 7; ++i)
    {
        auto mini_ret = node->play_mini(ps, exponent);
        if (i < 6)
            subpath.mini_node_seq[i] = static_cast<MiniNode *>(mini_ret.first);
        subpath.chosen_seq[i] = mini_ret.second;
        node = mini_ret.first;
    }
    SPDLOG_TRACE("Played one step.");
    return std::make_pair(static_cast<TreeNode *>(node), subpath);
}

pair<BaseNode *, ChosenChar> BaseNode::play_mini(PlayStrategy ps, float exponent) const
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

    // // Select the most visted.
    // auto it = std::max_element(action_counts.begin(), action_counts.end());
    // size_t index = std::distance(action_counts.begin(), it);

    // Select the node with the max return;
    size_t index;
    if (ps == PlayStrategy::MAX)
    {
        assert(max_index != -1);
        index = max_index;
    }
    else
    {
        auto probs = vec<float>();
        probs.reserve(action_counts.size());
        float sum = 0.0;
        if (ps == PlayStrategy::SAMPLE_AC)
        {
            for (size_t i = 0; i < action_counts.size(); ++i)
            {
                auto ac = action_counts[i];
                if (ac > 0)
                    probs.push_back(pruned[i] ? 1e-8 : pow(static_cast<float>(ac), exponent));
                else
                    probs.push_back(0);
                sum += probs.back();
            }
        }
        else if (ps == PlayStrategy::SAMPLE_MV)
        {
            for (size_t i = 0; i < max_values.size(); ++i)
            {
                auto mv = max_values[i];
                auto ac = action_counts[i];
                if (ac > 0)
                    if (pruned[i])
                        probs.push_back(1e-8);
                    else
                        probs.push_back(exp(mv * exponent));
                else
                    probs.push_back(0);
                sum += probs.back();
            }
        }
        for (auto &prob : probs)
            prob /= sum;

        float r = randf(1.0);
        float low = 0.0;
        float high = 0.0;
        index = 0;
        for (size_t i = 0; i < probs.size(); ++i)
        {
            high += probs[i];
            if ((r >= low) && (r < high))
            {
                index = i;
                break;
            }
            low = high;
        }
    }
    // // Select the node with the max average return, i.e., both puct_c and heur_c are set to 0.
    // auto scores = get_scores(0.0, 0.0);
    // auto it = std::max_element(scores.begin(), scores.end());
    // size_t index = std::distance(scores.begin(), it);

    // assert(!played);
    // played = true;

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

    // assert(!played);
    // // assert(!pruned[index]);
    // assert(children[index] != nullptr);
    // // assert(children[index]->num_unpruned_actions > 0);
    // played = true;
    // return children[index];

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

    // for (const auto ac : action_counts)
    // {
    //     probs.push_back(pow(static_cast<float>(ac), 10.0));
    //     sum += probs.back();
    // }
    // for (auto &prob : probs)
    //     prob /= sum;

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

    return std::make_pair(children[index], ChosenChar{index, permissible_chars[index]});
}

const IdSeq &TreeNode::get_id_seq(int order) const { return words[order]->id_seq; }

size_t TreeNode::size() const { return words.size(); }

size_t BaseNode::get_num_actions() const { return permissible_chars.size(); }

bool MiniNode::is_transitional() const { return false; }
bool TransitionNode::is_transitional() const { return true; }
bool TreeNode::is_transitional() const { return false; }
bool MiniNode::is_tree_node() const { return false; }
bool TreeNode::is_tree_node() const { return true; }

Trie<Word *, TreeNode *> TreeNode::t_table = Trie<Word *, TreeNode *>(nullptr);

TreeNode *TreeNode::get_tree_node(const vec<Word *> &words)
{
    auto new_node = new TreeNode(words);
    auto ret = new_node;
    if (TreeNode::t_table.get(words, ret))
        delete new_node;
    return ret;
}

TreeNode *TreeNode::get_tree_node(const vec<Word *> &words, bool stopped)
{
    auto new_node = new TreeNode(words, stopped);
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
}

void BaseNode::disconnect_from_children()
{
    for (size_t i = 0; i < children.size(); ++i)
    {
        const auto child = children[i];
        if (child != nullptr)
        {
            auto it = std::find(child->parents.begin(), child->parents.end(), this);
            assert(it != child->parents.end());
            auto index = std::distance(child->parents.begin(), it);
            child->parents.erase(it);
            child->parent_indices.erase(child->parent_indices.begin() + index);
            children[i] = nullptr;
        }
    }
}

size_t TreeNode::get_num_nodes() { return t_table.size(); }

void BaseNode::make_persistent() { persistent = true; }

BaseNode::~BaseNode()
{
    disconnect_from_parents();
    disconnect_from_children();
}

void TreeNode::remove_node_from_t_table(TreeNode *node)
{
    if (!node->stopped)
        TreeNode::t_table.remove(node->words);
}

bool BaseNode::is_persistent() const { return persistent; }

void BaseNode::connect(size_t index, BaseNode *child)
{
    if (children[index] == nullptr)
    {
        children[index] = child;
        child->parents.push_back(this);
        child->parent_indices.push_back(index);
    }
}

void BaseNode::init_edges()
{
    const size_t n = permissible_chars.size();
    children = vec<BaseNode *>(n, nullptr);
}

void BaseNode::update_stats(size_t index, float new_value, int game_count, float virtual_loss)
{
    action_counts[index] -= game_count - 1;
    if (action_counts[index] < 1)
    {
        std::cerr << index << '\n';
        std::cerr << action_counts[index] << '\n';
        assert(false);
    }
    // Update max value of the parent.
    if (new_value > max_value)
    {
        max_value = new_value;
        max_index = index;
    }
    if (new_value > max_values[index])
        max_values[index] = new_value;
    total_values[index] += game_count * virtual_loss + new_value;
    visit_count -= game_count - 1;
}

size_t BaseNode::get_action_index(abc_t action) const
{
    auto it = std::find(permissible_chars.begin(), permissible_chars.end(), action);
    if (it == permissible_chars.end())
        // {
        //     std::cerr << action << "\n";
        throw std::runtime_error("Target not found. This is usually the result of an action affecting zero site.");
    // }
    else
        return std::distance(permissible_chars.begin(), it);
}
const Affected &BaseNode::get_affected_at(size_t index) const { return affected[index]; }

void BaseNode::add_action(abc_t action, const Affected &affected)
{
    permissible_chars.push_back(action);
    this->affected.push_back(affected);
}

size_t BaseNode::get_num_affected_at(size_t index) const { return affected[index].size(); }

abc_t BaseNode::get_action_at(size_t index) const { return permissible_chars[index]; }

void BaseNode::update_affected_at(size_t index, int order, size_t pos, float misalign_score) { affected[index].push_back(order, pos, misalign_score); }

void BaseNode::init_stats()
{
    size_t n = permissible_chars.size();
    action_counts = vec<visit_t>(n, 0);
    total_values = vec<float>(n, 0.0);
    visit_count = 0;
    max_index = -1;
    max_value = -9999.9;
    max_values = vec<float>(n, -9999.9);
    // node->played = false;
}

inline void normalize(vec<float> &priors)
{
    float sum = 1e-8;
    for (const auto prior : priors)
        sum += prior;
    for (auto &prior : priors)
        prior /= sum;
}

inline vec<float> gather_priors(const vec<float> &values, const vec<abc_t> &indices)
{
    auto ret = vec<float>();
    ret.reserve(indices.size());
    for (const auto index : indices)
        ret.push_back(values[index]);
    normalize(ret);
    return ret;
}

void TreeNode::evaluate(const vec<vec<float>> &meta_priors, const vec<float> &special_priors)
{
    assert(is_expanded());
    if (is_evaluated())
        return;

    this->meta_priors = meta_priors;
    this->special_priors = special_priors;
    priors = gather_priors(meta_priors[0], permissible_chars);
}

void BaseNode::clear_priors() { priors.clear(); }

void MiniNode::evaluate()
{
    assert(is_expanded());
    if (is_evaluated())
        return;

    if (ap == ActionPhase::BEFORE) // NOTE(j_luo) Use `BEFORE` instead of `SPECIAL_TYPE` here.
        priors = base->evaluate_special_actions(permissible_chars);
    else
        priors = base->evaluate_actions(permissible_chars, ap);
}

const vec<abc_t> &BaseNode::get_actions() const { return permissible_chars; }
const vec<visit_t> &BaseNode::get_action_counts() const { return action_counts; }
const vec<float> &BaseNode::get_total_values() const { return total_values; }
const vec<float> &BaseNode::get_max_values() const { return max_values; }
visit_t BaseNode::get_visit_count() const { return visit_count; }
const vec<float> &BaseNode::get_priors() const { return priors; }

void BaseNode::virtual_select(size_t index, int game_count, float virtual_loss)
{
    action_counts[index] += game_count;
    total_values[index] -= game_count * virtual_loss;
    visit_count += game_count;
}

void BaseNode::init_pruned()
{
    size_t n = permissible_chars.size();
    num_unpruned_actions = n;
    pruned = vec<bool>(n, false);
}

const vec<bool> &BaseNode::get_pruned() const { return pruned; }

void BaseNode::dummy_evaluate() { priors = vec<float>(permissible_chars.size(), 0.0); }

void TransitionNode::init_rewards() { rewards = vec<float>(permissible_chars.size(), 0.0); }

float TransitionNode::get_reward_at(size_t index) const { return rewards[index]; }

void TransitionNode::set_reward_at(size_t index, float reward) { rewards[index] = reward; }

const vec<float> &TransitionNode::get_rewards() const { return rewards; }

void TreeNode::add_noise(const vec<vec<float>> &meta_noise, const vec<float> &special_noise, float noise_ratio)
{
    auto new_meta_priors = meta_priors;
    auto new_special_priors = special_priors;
    for (size_t i = 0; i < meta_noise.size(); ++i)
        for (size_t j = 0; j < meta_noise[i].size(); ++j)
            new_meta_priors[i][j] = new_meta_priors[i][j] * (1.0 - noise_ratio) + meta_noise[i][j] * noise_ratio;
    for (size_t i = 0; i < special_noise.size(); ++i)
        new_special_priors[i] = new_special_priors[i] * (1.0 - noise_ratio) + special_noise[i] * noise_ratio;
    evaluate(new_meta_priors, new_special_priors);
}

vec<float> TreeNode::evaluate_actions(const vec<abc_t> &actions, ActionPhase ap) const
{
    size_t index;
    switch (ap)
    {
    case ActionPhase::SPECIAL_TYPE:
        index = 1;
        break;
    case ActionPhase::AFTER:
        index = 2;
        break;
    case ActionPhase::PRE:
        index = 3;
        break;
    case ActionPhase::D_PRE:
        index = 4;
        break;
    case ActionPhase::POST:
        index = 5;
        break;
    }
    const auto &full_priors = meta_priors.at(index);
    // if (ap == ActionPhase::SPECIAL_TYPE)
    //     std::cerr << full_priors.at(56) << " " << full_priors.at(4) << "\n";
    return gather_priors(full_priors, actions);
}

vec<float> TreeNode::evaluate_special_actions(const vec<abc_t> &actions) const
{
    return gather_priors(special_priors, actions);
}

float TreeNode::get_dist() const { return dist; };

bool TreeNode::is_done() const { return done; };

pair<vec<vec<size_t>>, vec<vec<size_t>>> TreeNode::get_alignments() const
{
    auto ret = pair<vec<vec<size_t>>, vec<vec<size_t>>>();
    auto &almts1 = ret.first;
    auto &almts2 = ret.second;
    almts1.reserve(words.size());
    almts2.reserve(words.size());
    for (size_t i = 0; i < words.size(); ++i)
    {
        const auto &almt = words[i]->get_almt_at(i);
        almts1.push_back(almt.pos_seq1);
        almts2.push_back(almt.pos_seq2);
    }
    return ret;
}

void BaseNode::show_action_stats() const
{
    assert(is_expanded());
    for (size_t i = 0; i < permissible_chars.size(); ++i)
        std::cerr << permissible_chars[i] << ":" << affected[i].size() << " ";
    std::cerr << "\n";
}
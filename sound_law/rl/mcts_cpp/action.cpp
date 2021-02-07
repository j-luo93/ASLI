#include "action.hpp"

ActionSpace::ActionSpace(WordSpace *word_space, const ActionSpaceOpt &as_opt) : word_space(word_space), opt(as_opt) {}

TreeNode *ActionSpace::apply_new_action(TreeNode *node, const Subpath &subpath)
{
    // FIXME(j_luo) a bit repetive with env apply_action.
    MiniNode *last = subpath.mini_node_seq[4];
    abc_t after_id = subpath.chosen_seq[1].second;
    auto last_child_index = subpath.chosen_seq[5].first;
    TreeNode *new_node;
    if (subpath.stopped)
        new_node = new TreeNode(node->words, node->depth + 1, last, subpath.chosen_seq[5], true);
    else
    {
        auto new_words = vec<Word *>(node->words);
        const auto &aff = last->affected[last_child_index];
        // FIXME(j_luo) If everything is ordered, then perhaps we don't need hashing.
        auto order2pos = map<int, vec<size_t>>();
        for (const auto &item : aff)
            order2pos[item.first].push_back(item.second);
        for (const auto &item : order2pos)
        {
            auto order = item.first;
            auto new_id_seq = change_id_seq(node->words[order]->id_seq, item.second, after_id);
            auto new_word = word_space->get_word(new_id_seq);
            new_words[order] = new_word;
            word_space->set_edit_dist(new_word, order);
        }
        new_node = new TreeNode(new_words, node->depth + 1, last, subpath.chosen_seq[5], false);
        expand(new_node);
    }
    return new_node;
}

inline IdSeq ActionSpace::change_id_seq(const IdSeq &id_seq, const vec<size_t> &positions, abc_t after_id)
{
    // FIXME(j_luo)  add syncope and all special cases.
    auto new_id_seq = IdSeq(id_seq);
    for (const auto pos : positions)
        new_id_seq[pos] = after_id;
    if (after_id == opt.emp_id)
    {
        auto cleaned = IdSeq();
        cleaned.reserve(new_id_seq.size());
        for (const auto unit : new_id_seq)
            if (unit != opt.emp_id)
                cleaned.push_back(unit);
        return cleaned;
    }

    return new_id_seq;
}

void ActionSpace::register_permissible_change(abc_t before, abc_t after)
{
    permissible_changes[before].push_back(after);
}

Subpath ActionSpace::get_best_subpath(TreeNode *node, float puct_c, int game_count, float virtual_loss)
{
    SPDLOG_DEBUG("ActionSpace:: getting best subpath...");
    auto before = node->get_best_subaction(puct_c, game_count, virtual_loss);
    bool stopped = (before.first == 0);
    auto before_mn = get_mini_node(node, node, before, ActionPhase::BEFORE, stopped);
    if (expand(before_mn))
        evaluate(before_mn);
    SPDLOG_DEBUG("ActionSpace:: before done.");

    auto after = before_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto after_mn = get_mini_node(node, before_mn, after, ActionPhase::AFTER, stopped);
    if (expand(after_mn))
        evaluate(after_mn);
    SPDLOG_DEBUG("ActionSpace:: after done.");

    auto pre = after_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto pre_mn = get_mini_node(node, after_mn, pre, ActionPhase::PRE, stopped);
    if (expand(pre_mn))
        evaluate(pre_mn);
    SPDLOG_DEBUG("ActionSpace:: pre done.");

    auto d_pre = pre_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto d_pre_mn = get_mini_node(node, pre_mn, d_pre, ActionPhase::D_PRE, stopped);
    if (expand(d_pre_mn))
        evaluate(d_pre_mn);
    SPDLOG_DEBUG("ActionSpace:: d_pre done.");

    auto post = d_pre_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto post_mn = get_mini_node(node, d_pre_mn, post, ActionPhase::POST, stopped);
    if (expand(post_mn))
        evaluate(post_mn);
    SPDLOG_DEBUG("ActionSpace:: post done.");

    auto d_post = post_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    SPDLOG_DEBUG("ActionSpace:: d_post done.");

    Subpath subpath;
    subpath.chosen_seq = {before, after, pre, d_pre, post, d_post};
    subpath.mini_node_seq = {before_mn, after_mn, pre_mn, d_pre_mn, post_mn};
    subpath.stopped = stopped;
    return subpath;
}

MiniNode *ActionSpace::get_mini_node(TreeNode *base, BaseNode *parent, const ChosenChar &chosen, ActionPhase ap, bool stopped)
{
    BaseNode *&child = parent->children[chosen.first];
    bool is_transition = (ap == ActionPhase::POST);
    if (child == nullptr)
        if (is_transition)
            child = new TransitionNode(base, static_cast<MiniNode *>(parent), chosen, stopped);
        else
            child = new MiniNode(base, parent, chosen, ap, stopped);
    if (is_transition)
        return static_cast<TransitionNode *>(child);
    else
        return static_cast<MiniNode *>(child);
}

void ActionSpace::expand(TreeNode *node)
{
    std::lock_guard<std::mutex> lock(node->mtx);
    SPDLOG_DEBUG("ActionSpace:: expanding node...");

    if (node->is_expanded())
    {
        SPDLOG_DEBUG("ActionSpace:: node already expanded.");
        return;
    }

    // Null/Stop option.
    node->permissible_chars.push_back(opt.null_id);
    node->affected = vec<Affected>({{}});

    auto char_map = map<abc_t, size_t>();
    for (int order = 0; order < node->words.size(); ++order)
    {
        auto &id_seq = node->words[order]->id_seq;
        size_t n = id_seq.size();
        // Skip the boundaries.
        for (int pos = 1; pos < n - 1; ++pos)
            update_affected(node, id_seq, order, pos, 0, char_map);
    }

    clear_stats(node);
    SPDLOG_DEBUG("ActionSpace:: node expanded with #actions {}.", node->permissible_chars.size());

    // for (size_t i = 0; i < node->permissible_chars.size(); ++i)
    // {
    //     std::cerr << "unit: " << node->permissible_chars[i] << "\n";
    //     std::cerr << "affected " << node->affected[i].size() << "\n";
    //     for (const auto &item : node->affected[i])
    //     {
    //         std::cerr << item.first << ":" << item.second << " ";
    //     }
    //     std::cerr << "\n";
    // }
}

bool ActionSpace::expand(MiniNode *node)
{
    std::lock_guard<std::mutex> lock(node->mtx);

    SPDLOG_TRACE("MiniNode expanded already.");
    if (node->is_expanded())
        return false;

    SPDLOG_TRACE("ActionSpace:: Expanding {0}, chosen_char: ({1}, {2}), parent #actions {3}",
                 str::from(node->ap), node->chosen_char.first, node->chosen_char.second, node->parent->permissible_chars.size());

    if (node->stopped)
    {
        node->permissible_chars.push_back(opt.null_id);
        node->affected = vec<Affected>({{}});
    }
    // Only BEFORE doesn't have NONE as an option (if not stopped).
    else if (node->ap != ActionPhase::BEFORE)
    {
        node->permissible_chars.push_back(opt.null_id);
        // Affected positions will not be further narrowed down.
        assert(node->parent != nullptr);
        node->affected = vec<Affected>({node->parent->affected[node->chosen_char.first]});
    }

    if (node->stopped)
        SPDLOG_TRACE("Phase {}, keeping only Null action due to stopped status.", str::from(node->ap));
    // If pre_id is null then d_pre_id should also be null. Same goes for post_id and d_post_id.
    else if (((node->ap == ActionPhase::PRE) || (node->ap == ActionPhase::POST)) && (node->chosen_char.second == opt.null_id))
        SPDLOG_TRACE("Phase {}, keeping only Null action.", str::from(node->ap));

    // For the five intermediate mini nodes:
    // before: after_ids based on main's affected and before's chosen_char
    // after: pre_ids based on before's affected
    // pre: d_pre_ids based on after's affected and chosen_char
    // d_pre: post_ids based on pre's affected and chosen_char
    // post: d_post_ids based on d_pre's affected and chosen_char
    else if (node->ap == ActionPhase::BEFORE)
    {
        // FIXME(j_luo) This is not very efficient.
        assert(node->parent != nullptr);
        node->permissible_chars = permissible_changes[node->chosen_char.second];
        node->affected = vec<Affected>(node->permissible_chars.size(), node->parent->affected[node->chosen_char.first]);
    }
    else
    {
        int offset;
        switch (node->ap)
        {
        case ActionPhase::AFTER:
            offset = -1;
            break;
        case ActionPhase::PRE:
            offset = -2;
            break;
        case ActionPhase::D_PRE:
            offset = 1;
            break;
        case ActionPhase::POST:
            offset = 2;
            break;
        }
        auto &words = node->base->words;
        auto &affected = node->parent->affected[node->chosen_char.first];
        auto char_map = map<abc_t, size_t>();
        for (const auto &aff : affected)
        {
            int order = aff.first;
            update_affected(node, words[order]->id_seq, order, aff.second, offset, char_map);
        }
    }
    clear_stats(node);
    SPDLOG_DEBUG("ActionSpace:: mini node expanded with #actions {}.", node->permissible_chars.size());
    // for (size_t i = 0; i < node->permissible_chars.size(); ++i)
    // {
    //     std::cerr << "unit: " << node->permissible_chars[i] << "\n";
    //     std::cerr << "affected " << node->affected[i].size() << "\n";
    //     for (const auto &item : node->affected[i])
    //     {
    //         std::cerr << item.first << ":" << item.second << " ";
    //     }
    //     std::cerr << "\n";
    // }
    return true;
}

void ActionSpace::update_affected(BaseNode *node, const IdSeq &id_seq, int order, size_t pos, int offset, map<abc_t, size_t> &char_map)
{
    int new_pos = pos + offset;
    if ((new_pos < 0) || (new_pos >= id_seq.size()))
        return;

    auto unit = id_seq[new_pos];
    if (!char_map.contains(unit))
    {
        // Add one more permission char.
        char_map[unit] = node->permissible_chars.size();
        node->permissible_chars.push_back(unit);
        node->affected.push_back(Affected({{order, pos}}));
    }
    else
    {
        // Add one more position.
        auto &aff = node->affected[char_map[unit]];
        aff.push_back({order, pos});
    }
}

void ActionSpace::evaluate(MiniNode *node)
{
    assert(node->is_expanded());
    auto &full_priors = node->base->meta_priors[static_cast<int>(node->ap) + 1]; // +1 because the first is used for the tree node.
    node->priors.clear();
    for (const auto unit : node->permissible_chars)
        node->priors.push_back(full_priors[unit]);
}

void ActionSpace::clear_stats(BaseNode *node)
{
    size_t n = node->permissible_chars.size();
    node->children = vec<BaseNode *>(n, nullptr);
    node->action_counts = vec<visit_t>(n, 0);
    node->total_values = vec<float>(n, 0.0);
    auto tnode = dynamic_cast<TransitionNode *>(node);
    if (tnode != nullptr)
        tnode->rewards = vec<float>(n, 0.0);
}

void ActionSpace::evaluate(TreeNode *node, const MetaPriors &meta_priors)
{
    assert(node->is_expanded());
    node->meta_priors = meta_priors;
    node->priors.clear();
    auto &full_priors = node->meta_priors[0];
    for (const auto unit : node->permissible_chars)
        node->priors.push_back(full_priors[unit]);
}
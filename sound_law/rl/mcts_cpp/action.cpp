#include "action.hpp"

ActionSpace::ActionSpace(WordSpace *word_space, const ActionSpaceOpt &as_opt) : word_space(word_space), opt(as_opt) {}

TreeNode *ActionSpace::apply_new_action(TreeNode *node, const Subpath &subpath)
{
    // FIXME(j_luo) a bit repetive with env apply_action.
    MiniNode *last = subpath.mini_node_seq[5];
    abc_t after_id = subpath.chosen_seq[1].second;
    auto last_child_index = subpath.chosen_seq[6].first;
    TreeNode *new_node;
    if (subpath.stopped)
        new_node = new TreeNode(node->words, node->depth + 1, last, subpath.chosen_seq[6], true);
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
            word_space->set_edit_dist_at(new_word, order);
        }
        new_node = new TreeNode(new_words, node->depth + 1, last, subpath.chosen_seq[6], false);
        expand(new_node);
    }
    return new_node;
}

inline int get_index(abc_t target, const vec<abc_t> &chars)
{
    for (int i = 0; i < chars.size(); ++i)
        if (chars[i] == target)
            return i;
    // std::cerr << target << "\n";
    // for (const auto c : chars)
    //     std::cerr << c << " ";
    // assert(false);
    throw std::runtime_error("Target not found.");
}

inline void dummy_evaluate(vec<float> &priors, size_t size) { priors = vec<float>(size, 0.0); }

TreeNode *ActionSpace::apply_action(TreeNode *node,
                                    abc_t before_id,
                                    abc_t after_id,
                                    abc_t pre_id,
                                    abc_t d_pre_id,
                                    abc_t post_id,
                                    abc_t d_post_id,
                                    SpecialType st)
{
    // std::cerr << "before\n";
    auto before = ChosenChar({get_index(before_id, node->permissible_chars), before_id});
    bool stopped = (before.first == 0);
    auto before_mn = get_mini_node(node, node, before, ActionPhase::BEFORE, stopped);
    if (expand(before_mn, false))
        dummy_evaluate(before_mn->priors, before_mn->permissible_chars.size());

    // std::cerr << "after\n";
    auto after = ChosenChar({get_index(after_id, before_mn->permissible_chars), after_id});
    auto after_mn = get_mini_node(node, before_mn, after, ActionPhase::AFTER, stopped);
    if (expand(after_mn, false))
        dummy_evaluate(after_mn->priors, after_mn->permissible_chars.size());

    // std::cerr << "special_type\n";
    abc_t st_abc = static_cast<abc_t>(st);
    bool use_vowel_seq = (st == SpecialType::VS);
    auto special_type = ChosenChar({get_index(st_abc, after_mn->permissible_chars), st_abc});
    auto st_mn = get_mini_node(node, after_mn, special_type, ActionPhase::SPECIAL_TYPE, stopped);
    if (expand(st_mn, use_vowel_seq))
        dummy_evaluate(st_mn->priors, st_mn->permissible_chars.size());

    // std::cerr << "pre\n";
    auto pre = ChosenChar({get_index(pre_id, st_mn->permissible_chars), pre_id});
    auto pre_mn = get_mini_node(node, st_mn, pre, ActionPhase::PRE, stopped);
    if (expand(pre_mn, use_vowel_seq))
        dummy_evaluate(pre_mn->priors, pre_mn->permissible_chars.size());

    auto d_pre = ChosenChar({get_index(d_pre_id, pre_mn->permissible_chars), d_pre_id});
    // std::cerr << "d_pre\n";
    auto d_pre_mn = get_mini_node(node, pre_mn, d_pre, ActionPhase::D_PRE, stopped);
    if (expand(d_pre_mn, use_vowel_seq))
        dummy_evaluate(d_pre_mn->priors, d_pre_mn->permissible_chars.size());

    // std::cerr << "post\n";
    auto post = ChosenChar({get_index(post_id, d_pre_mn->permissible_chars), post_id});
    auto post_mn = get_mini_node(node, d_pre_mn, post, ActionPhase::POST, stopped);
    if (expand(post_mn, use_vowel_seq))
        dummy_evaluate(post_mn->priors, post_mn->permissible_chars.size());

    // std::cerr << "d_post\n";
    auto d_post = ChosenChar({get_index(d_post_id, post_mn->permissible_chars), d_post_id});

    Subpath subpath;
    subpath.chosen_seq = {before, after, special_type, pre, d_pre, post, d_post};
    subpath.mini_node_seq = {before_mn, after_mn, st_mn, pre_mn, d_pre_mn, post_mn};
    subpath.stopped = stopped;

    // FIXME(j_luo) This shouldn't create a new node.
    return apply_new_action(node, subpath);
}

inline IdSeq ActionSpace::change_id_seq(const IdSeq &id_seq, const vec<size_t> &positions, abc_t after_id)
{
    // FIXME(j_luo)   add all special cases.
    auto new_id_seq = IdSeq(id_seq);
    auto stressed_after_id = word_space->opt.unit2stressed[after_id];
    auto unstressed_after_id = word_space->opt.unit2unstressed[after_id];
    for (const auto pos : positions)
    {
        auto stress = word_space->opt.unit_stress[new_id_seq[pos]];
        switch (stress)
        {
        case Stress::NOSTRESS:
            new_id_seq[pos] = after_id;
            break;
        case Stress::STRESSED:
            new_id_seq[pos] = stressed_after_id;
            break;
        case Stress::UNSTRESSED:
            new_id_seq[pos] = unstressed_after_id;
            break;
        }
    }
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
    if (expand(before_mn, false))
        evaluate(before_mn);
    SPDLOG_DEBUG("ActionSpace:: before done.");

    auto after = before_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto after_mn = get_mini_node(node, before_mn, after, ActionPhase::AFTER, stopped);
    if (expand(after_mn, false))
        evaluate(after_mn);
    SPDLOG_DEBUG("ActionSpace:: after done.");

    bool use_vowel_seq = (static_cast<SpecialType>(after_mn->chosen_char.second) == SpecialType::VS);
    auto special_type = after_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto st_mn = get_mini_node(node, after_mn, special_type, ActionPhase::SPECIAL_TYPE, stopped);
    if (expand(st_mn, use_vowel_seq))
        evaluate(st_mn);
    SPDLOG_DEBUG("ActionSpace:: special_type done.");

    auto pre = st_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto pre_mn = get_mini_node(node, st_mn, pre, ActionPhase::PRE, stopped);
    if (expand(pre_mn, use_vowel_seq))
        evaluate(pre_mn);
    SPDLOG_DEBUG("ActionSpace:: pre done.");

    auto d_pre = pre_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto d_pre_mn = get_mini_node(node, pre_mn, d_pre, ActionPhase::D_PRE, stopped);
    if (expand(d_pre_mn, use_vowel_seq))
        evaluate(d_pre_mn);
    SPDLOG_DEBUG("ActionSpace:: d_pre done.");

    auto post = d_pre_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    auto post_mn = get_mini_node(node, d_pre_mn, post, ActionPhase::POST, stopped);
    if (expand(post_mn, use_vowel_seq))
        evaluate(post_mn);
    SPDLOG_DEBUG("ActionSpace:: post done.");

    auto d_post = post_mn->get_best_subaction(puct_c, game_count, virtual_loss);
    SPDLOG_DEBUG("ActionSpace:: d_post done.");

    Subpath subpath;
    subpath.chosen_seq = {before, after, special_type, pre, d_pre, post, d_post};
    subpath.mini_node_seq = {before_mn, after_mn, st_mn, pre_mn, d_pre_mn, post_mn};
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

inline bool in_bound(int pos, size_t max) { return ((pos >= 0) && (pos < max)); }

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
            if (in_bound(pos, n))
                update_affected(node, id_seq[pos], order, pos, char_map);
    }

    clear_stats(node);
    SPDLOG_DEBUG("ActionSpace:: node expanded with #actions {}.", node->permissible_chars.size());

    // std::cerr << "-----------------\n";
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

void ActionSpace::expand_before(MiniNode *node)
{
    // FIXME(j_luo) This is not very efficient.
    node->permissible_chars = permissible_changes[node->chosen_char.second];
    node->affected = vec<Affected>(node->permissible_chars.size(), node->parent->affected[node->chosen_char.first]);
}
void ActionSpace::expand_after(MiniNode *node)
{
    auto unit = node->parent->chosen_char.second;
    node->permissible_chars.push_back(static_cast<abc_t>(SpecialType::NONE));
    if (word_space->opt.is_vowel[unit])
        node->permissible_chars.push_back(static_cast<abc_t>(SpecialType::VS));
    node->affected = vec<Affected>(node->permissible_chars.size(), node->parent->affected[node->chosen_char.first]);
}
void ActionSpace::expand_normal(MiniNode *node, int offset, bool use_vowel_seq)
{
    expand_null(node);

    auto &words = node->base->words;
    auto &affected = node->parent->affected[node->chosen_char.first];
    auto char_map = map<abc_t, size_t>();
    for (const auto &aff : affected)
    {
        int order = aff.first;
        auto word = words[order];
        if (use_vowel_seq)
        {
            // Get the position/index in the vowel seq.
            auto vowel_pos = word->id2vowel[aff.second] + offset;
            auto &vowel_seq = words[order]->vowel_seq;
            if (in_bound(vowel_pos, vowel_seq.size()))
                update_affected(node, vowel_seq[vowel_pos], order, aff.second, char_map);
        }
        else
        {
            auto pos = aff.second + offset;
            auto &id_seq = words[order]->id_seq;
            if (in_bound(pos, id_seq.size()))
                update_affected(node, id_seq[pos], order, aff.second, char_map);
        }
    }
}
bool ActionSpace::expand_null_only(MiniNode *node)
{
    if (node->chosen_char.second == opt.null_id)
    {
        SPDLOG_TRACE("Phase {}, keeping only Null action.", str::from(node->ap));
        expand_null(node);
        return true;
    }
    return false;
}
void ActionSpace::expand_special_type(MiniNode *node, bool use_vowel_seq) { expand_normal(node, -1, use_vowel_seq); }
void ActionSpace::expand_pre(MiniNode *node, bool use_vowel_seq)
{
    if (!expand_null_only(node))
        expand_normal(node, -2, use_vowel_seq);
}
void ActionSpace::expand_d_pre(MiniNode *node, bool use_vowel_seq) { expand_normal(node, 1, use_vowel_seq); }
void ActionSpace::expand_post(MiniNode *node, bool use_vowel_seq)
{
    if (!expand_null_only(node))
        expand_normal(node, 2, use_vowel_seq);
}
void ActionSpace::expand_null(MiniNode *node)
{
    node->permissible_chars.push_back(opt.null_id);
    // Affected positions will not be further narrowed down.
    node->affected = vec<Affected>({node->parent->affected[node->chosen_char.first]});
}

bool ActionSpace::expand(MiniNode *node, bool use_vowel_seq)
{
    std::lock_guard<std::mutex> lock(node->mtx);

    if (node->is_expanded())
    {
        SPDLOG_TRACE("MiniNode expanded already.");
        return false;
    }

    SPDLOG_TRACE("ActionSpace:: Expanding {0}, chosen_char: ({1}, {2}), parent #actions {3}",
                 str::from(node->ap), node->chosen_char.first, node->chosen_char.second, node->parent->permissible_chars.size());

    if (node->stopped)
    {
        node->permissible_chars.push_back(opt.null_id);
        node->affected = vec<Affected>({{}});
        SPDLOG_TRACE("Phase {}, keeping only Null action due to stopped status.", str::from(node->ap));
    }
    else
        switch (node->ap)
        {
        case ActionPhase::BEFORE:
            expand_before(node);
            break;
        case ActionPhase::AFTER:
            expand_after(node);
            break;
        case ActionPhase::SPECIAL_TYPE:
            expand_special_type(node, use_vowel_seq);
            break;
        case ActionPhase::PRE:
            expand_pre(node, use_vowel_seq);
            break;
        case ActionPhase::D_PRE:
            expand_d_pre(node, use_vowel_seq);
            break;
        case ActionPhase::POST:
            expand_post(node, use_vowel_seq);
            break;
        }

    clear_stats(node);
    SPDLOG_DEBUG("ActionSpace:: mini node expanded with #actions {}.", node->permissible_chars.size());
    return true;

    // // Only BEFORE doesn't have NONE as an option (if not stopped).
    // else if (node->ap != ActionPhase::BEFORE)
    // {
    //     node->permissible_chars.push_back(opt.null_id);
    //     // Affected positions will not be further narrowed down.
    //     assert(node->parent != nullptr);
    //     node->affected = vec<Affected>({node->parent->affected[node->chosen_char.first]});
    // }

    // // If pre_id is null then d_pre_id should also be null. Same goes for post_id and d_post_id.
    // else if (((node->ap == ActionPhase::PRE) || (node->ap == ActionPhase::POST)) && (node->chosen_char.second == opt.null_id))
    //     SPDLOG_TRACE("Phase {}, keeping only Null action.", str::from(node->ap));

    // For the five intermediate mini nodes:
    // before: after_ids based on main's affected and before's chosen_char
    // after: pre_ids based on before's affected
    // pre: d_pre_ids based on after's affected and chosen_char
    // d_pre: post_ids based on pre's affected and chosen_char
    // post: d_post_ids based on d_pre's affected and chosen_char
    // else if (node->ap == ActionPhase::BEFORE)
    // {
    //     // FIXME(j_luo) This is not very efficient.
    //     assert(node->parent != nullptr);
    //     node->permissible_chars = permissible_changes[node->chosen_char.second];
    //     node->affected = vec<Affected>(node->permissible_chars.size(), node->parent->affected[node->chosen_char.first]);
    // }
    // else
    // {
    //     int offset;
    //     switch (node->ap)
    //     {
    //     case ActionPhase::AFTER:
    //         offset = -1;
    //         break;
    //     case ActionPhase::PRE:
    //         offset = -2;
    //         break;
    //     case ActionPhase::D_PRE:
    //         offset = 1;
    //         break;
    //     case ActionPhase::POST:
    //         offset = 2;
    //         break;
    //     }
    //     auto &words = node->base->words;
    //     auto &affected = node->parent->affected[node->chosen_char.first];
    //     auto char_map = map<abc_t, size_t>();
    //     for (const auto &aff : affected)
    //     {
    //         int order = aff.first;
    //         update_affected(node, words[order]->id_seq, order, aff.second, offset, char_map);
    //     }
    // }

    // std::cerr << "-----------------\n";
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

void ActionSpace::update_affected(BaseNode *node, abc_t unit, int order, size_t pos, map<abc_t, size_t> &char_map)
{
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

    if (word_space->opt.unit_stress[unit] != Stress::NOSTRESS)
        update_affected(node, word_space->opt.unit2base[unit], order, pos, char_map);
}

void ActionSpace::evaluate(MiniNode *node)
{
    assert(node->is_expanded());
    node->priors.clear();
    node->priors.reserve(node->permissible_chars.size());
    if (node->ap == ActionPhase::SPECIAL_TYPE)
    {
        for (const auto st : node->permissible_chars)
            node->priors.push_back(node->base->special_priors[st]);
    }
    else
    {
        auto &full_priors = node->base->meta_priors[static_cast<int>(node->ap) + 1]; // +1 because the first is used for the tree node.
        for (const auto unit : node->permissible_chars)
            node->priors.push_back(full_priors[unit]);
    }
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

void ActionSpace::evaluate(TreeNode *node, const MetaPriors &meta_priors, const vec<float> &special_priors)
{
    assert(node->is_expanded());
    node->meta_priors = meta_priors;
    node->special_priors = special_priors;
    node->priors.clear();
    auto &full_priors = node->meta_priors[0];
    for (const auto unit : node->permissible_chars)
        node->priors.push_back(full_priors[unit]);
}
#include "action.hpp"

ActionSpace::ActionSpace(WordSpace *word_space, const ActionSpaceOpt &as_opt) : word_space(word_space), opt(as_opt) {}

TreeNode *ActionSpace::apply_new_action(TreeNode *node, const Subpath &subpath)
{
    // FIXME(j_luo) a bit repetive with env apply_action.
    MiniNode *last = subpath.mini_node_seq[5];
    abc_t after_id = subpath.chosen_seq[2].second;
    auto last_child_index = subpath.chosen_seq[6].first;
    SpecialType st = static_cast<SpecialType>(subpath.chosen_seq[1].second);
    TreeNode *new_node;
    if (subpath.stopped)
    // A new node should always be created for STOP.
    {
        new_node = NodeFactory::get_stopped_node(node);
        EdgeBuilder::connect(last, last_child_index, new_node);
    }
    else
    {
        auto new_words = vec<Word *>(node->words);
        const auto &aff = last->get_affected_at(last_child_index);
        // FIXME(j_luo) If everything is ordered, then perhaps we don't need hashing.
        auto order2pos = map<int, vec<size_t>>();
        for (const auto &item : aff)
            order2pos[item.first].push_back(item.second);
        for (const auto &item : order2pos)
        {
            auto order = item.first;
            auto new_id_seq = change_id_seq(node->words[order]->id_seq, item.second, after_id, st);
            auto new_word = word_space->get_word(new_id_seq);
            new_words[order] = new_word;
            word_space->set_edit_dist_at(new_word, order);
        }
        new_node = NodeFactory::get_tree_node(new_words, false);
        EdgeBuilder::connect(last, last_child_index, new_node);
        expand(new_node);
        if ((node->get_dist() - new_node->get_dist()) < opt.dist_threshold)
            PruningManager::prune(last, last_child_index);
        // new_node->prune();
    }
    return new_node;
}

TreeNode *ActionSpace::apply_action(TreeNode *node,
                                    abc_t before_id,
                                    abc_t after_id,
                                    abc_t pre_id,
                                    abc_t d_pre_id,
                                    abc_t post_id,
                                    abc_t d_post_id,
                                    SpecialType st)
{
    Subpath subpath = Subpath();

    // std::cerr << "before\n";
    auto before = ChosenChar({node->get_action_index(before_id), before_id});
    bool stopped = (before.first == 0);
    auto before_mn = get_mini_node(node, node, before, ActionPhase::BEFORE, stopped);
    subpath.chosen_seq[0] = before;
    subpath.mini_node_seq[0] = before_mn;
    subpath.stopped = stopped;
    expand(before_mn, subpath, false, false);
    ActionManager::dummy_evaluate(before_mn);

    // std::cerr << "special_type\n";
    abc_t st_abc = static_cast<abc_t>(st);
    auto special_type = ChosenChar({before_mn->get_action_index(st_abc), st_abc});
    auto st_mn = get_mini_node(node, before_mn, special_type, ActionPhase::SPECIAL_TYPE, stopped);
    subpath.chosen_seq[1] = special_type;
    subpath.mini_node_seq[1] = st_mn;
    expand(st_mn, subpath, false, true);
    ActionManager::dummy_evaluate(st_mn);

    // std::cerr << "after\n";
    // std::cerr << after_id << "\n";
    // for (const auto x : st_mn->permissible_chars)
    //     std::cerr << x << " ";
    // std::cerr << "\n";
    auto after = ChosenChar({st_mn->get_action_index(after_id), after_id});
    auto after_mn = get_mini_node(node, st_mn, after, ActionPhase::AFTER, stopped);
    bool use_vowel_seq = (st == SpecialType::VS);
    subpath.chosen_seq[2] = after;
    subpath.mini_node_seq[2] = after_mn;
    expand(after_mn, subpath, use_vowel_seq, false);
    ActionManager::dummy_evaluate(after_mn);

    // std::cerr << "pre\n";
    auto pre = ChosenChar({after_mn->get_action_index(pre_id), pre_id});
    auto pre_mn = get_mini_node(node, after_mn, pre, ActionPhase::PRE, stopped);
    subpath.chosen_seq[3] = pre;
    subpath.mini_node_seq[3] = pre_mn;
    expand(pre_mn, subpath, use_vowel_seq, false);
    ActionManager::dummy_evaluate(pre_mn);

    auto d_pre = ChosenChar({pre_mn->get_action_index(d_pre_id), d_pre_id});
    // std::cerr << "d_pre\n";
    auto d_pre_mn = get_mini_node(node, pre_mn, d_pre, ActionPhase::D_PRE, stopped);
    subpath.chosen_seq[4] = d_pre;
    subpath.mini_node_seq[4] = d_pre_mn;
    expand(d_pre_mn, subpath, use_vowel_seq, false);
    ActionManager::dummy_evaluate(d_pre_mn);

    // std::cerr << "post\n";
    auto post = ChosenChar({d_pre_mn->get_action_index(post_id), post_id});
    auto post_mn = get_mini_node(node, d_pre_mn, post, ActionPhase::POST, stopped);
    subpath.chosen_seq[5] = post;
    subpath.mini_node_seq[5] = post_mn;
    expand(post_mn, subpath, use_vowel_seq, false);
    ActionManager::dummy_evaluate(post_mn);

    // std::cerr << "d_post\n";
    auto d_post = ChosenChar({post_mn->get_action_index(d_post_id), d_post_id});
    subpath.chosen_seq[6] = d_post;

    // FIXME(j_luo) This shouldn't create a new node.
    return apply_new_action(node, subpath);
}

inline IdSeq ActionSpace::change_id_seq(const IdSeq &id_seq, const vec<size_t> &positions, abc_t after_id, SpecialType st)
{
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
    if (st == SpecialType::CLL)
    {
        for (const auto pos : positions)
        {
            assert(pos > 0);
            new_id_seq[pos - 1] = opt.emp_id;
        }
    }
    else if (st == SpecialType::CLR)
    {
        for (const auto pos : positions)
        {
            assert(pos < (id_seq.size() - 1));
            new_id_seq[pos + 1] = opt.emp_id;
        }
    }

    if ((after_id == opt.emp_id) || (st == SpecialType::CLL) || (st == SpecialType::CLR))
    {
        auto cleaned = IdSeq();
        cleaned.reserve(new_id_seq.size());
        for (const auto unit : new_id_seq)
            if (unit != opt.emp_id)
                cleaned.push_back(unit);
        return cleaned;
    }

    if ((st == SpecialType::GBJ) || (st == SpecialType::GBW))
    {
        abc_t glide = ((st == SpecialType::GBJ) ? opt.glide_j : opt.glide_w);
        auto to_insert = vec<bool>(new_id_seq.size(), false);
        for (const auto pos : positions)
            to_insert[pos] = true;
        auto inserted = IdSeq();
        inserted.reserve(new_id_seq.size() * 10);
        for (size_t i = 0; i < new_id_seq.size(); ++i)
        {
            if (to_insert[i])
                inserted.push_back(glide);
            inserted.push_back(new_id_seq[i]);
        }
        return inserted;
    }

    return new_id_seq;
}

void ActionSpace::register_permissible_change(abc_t before, abc_t after) { permissible_changes[before].push_back(after); }
void ActionSpace::register_cl_map(abc_t before, abc_t after) { cl_map[before] = after; }
void ActionSpace::register_gbj_map(abc_t before, abc_t after) { gbj_map[before] = after; }
void ActionSpace::register_gbw_map(abc_t before, abc_t after) { gbw_map[before] = after; }

Subpath ActionSpace::get_best_subpath(TreeNode *node, float puct_c, int game_count, float virtual_loss, float heur_c, bool add_noise) const
{
    Subpath subpath = Subpath();

    // std::cerr << "before\n";
    SPDLOG_DEBUG("ActionSpace:: getting best subpath...");
    auto before = node->get_best_action(puct_c, heur_c, add_noise);
    bool stopped = (before.first == 0);
    auto before_mn = get_mini_node(node, node, before, ActionPhase::BEFORE, stopped);
    subpath.chosen_seq[0] = before;
    subpath.mini_node_seq[0] = before_mn;
    subpath.stopped = stopped;
    expand(before_mn, subpath, false, false);
    evaluate(before_mn);
    SPDLOG_DEBUG("ActionSpace:: before done.");

    // std::cerr << "special_type\n";
    auto special_type = before_mn->get_best_action(puct_c, heur_c, add_noise);
    auto st_mn = get_mini_node(node, before_mn, special_type, ActionPhase::SPECIAL_TYPE, stopped);
    subpath.chosen_seq[1] = special_type;
    subpath.mini_node_seq[1] = st_mn;
    expand(st_mn, subpath, false, false);
    evaluate(st_mn);
    SPDLOG_DEBUG("ActionSpace:: special_type done.");

    // std::cerr << "after\n";
    auto after = st_mn->get_best_action(puct_c, heur_c, add_noise);
    bool use_vowel_seq = (static_cast<SpecialType>(special_type.second) == SpecialType::VS);
    auto after_mn = get_mini_node(node, st_mn, after, ActionPhase::AFTER, stopped);
    subpath.chosen_seq[2] = after;
    subpath.mini_node_seq[2] = after_mn;
    expand(after_mn, subpath, use_vowel_seq, false);
    evaluate(after_mn);
    SPDLOG_DEBUG("ActionSpace:: after done.");

    // std::cerr << "pre\n";
    auto pre = after_mn->get_best_action(puct_c, heur_c, add_noise);
    auto pre_mn = get_mini_node(node, after_mn, pre, ActionPhase::PRE, stopped);
    subpath.chosen_seq[3] = pre;
    subpath.mini_node_seq[3] = pre_mn;
    expand(pre_mn, subpath, use_vowel_seq, false);
    evaluate(pre_mn);
    SPDLOG_DEBUG("ActionSpace:: pre done.");

    // std::cerr << "d_pre\n";
    auto d_pre = pre_mn->get_best_action(puct_c, heur_c, add_noise);
    auto d_pre_mn = get_mini_node(node, pre_mn, d_pre, ActionPhase::D_PRE, stopped);
    subpath.chosen_seq[4] = d_pre;
    subpath.mini_node_seq[4] = d_pre_mn;
    expand(d_pre_mn, subpath, use_vowel_seq, false);
    evaluate(d_pre_mn);
    SPDLOG_DEBUG("ActionSpace:: d_pre done.");

    // std::cerr << "post\n";
    auto post = d_pre_mn->get_best_action(puct_c, heur_c, add_noise);
    auto post_mn = get_mini_node(node, d_pre_mn, post, ActionPhase::POST, stopped);
    subpath.chosen_seq[5] = post;
    subpath.mini_node_seq[5] = post_mn;
    expand(post_mn, subpath, use_vowel_seq, false);
    evaluate(post_mn);
    SPDLOG_DEBUG("ActionSpace:: post done.");

    // std::cerr << "d_post\n";
    auto d_post = post_mn->get_best_action(puct_c, heur_c, add_noise);
    subpath.chosen_seq[6] = d_post;
    SPDLOG_DEBUG("ActionSpace:: d_post done.");

    connect(node, subpath);
    return subpath;
}

MiniNode *ActionSpace::get_mini_node(TreeNode *base, BaseNode *parent, const ChosenChar &chosen, ActionPhase ap, bool stopped) const
{
    // BaseNode *&child = parent->children[chosen.first];
    bool is_transition = (ap == ActionPhase::POST);
    BaseNode *child;
    if (!parent->has_child(chosen.first))
    {
        if (is_transition)
            child = NodeFactory::get_transition_node(base, stopped);
        else
            child = NodeFactory::get_mini_node(base, ap, stopped);
        EdgeBuilder::connect(parent, chosen.first, child);
    }
    else
        child = parent->get_child(chosen.first);
    if (is_transition)
        return static_cast<TransitionNode *>(child);
    else
        return static_cast<MiniNode *>(child);
}

inline bool in_bound(int pos, size_t max) { return ((pos >= 0) && (pos < max)); }

void ActionSpace::expand(TreeNode *node) const
{
    SPDLOG_DEBUG("ActionSpace:: expanding node...");

    if (node->is_expanded())
    {
        SPDLOG_DEBUG("ActionSpace:: node already expanded.");
        return;
    }

    // Null/Stop option.
    ActionManager::add_action(node, opt.null_id, Affected());

    auto char_map = map<abc_t, size_t>();
    for (int order = 0; order < node->words.size(); ++order)
    {
        auto &id_seq = node->words[order]->id_seq;
        size_t n = id_seq.size();
        // Skip the boundaries.
        for (int pos = 1; pos < n - 1; ++pos)
            if (in_bound(pos, n))
                update_affected(node, id_seq[pos], order, pos, char_map, false);
    }

    expand_stats(node);
    SPDLOG_DEBUG("ActionSpace:: node expanded with #actions {}.", node->get_num_actions());

    // Skip STOP.
    for (size_t i = 1; i < node->get_num_actions(); ++i)
        if (node->get_num_affected_at(i) < opt.site_threshold)
            PruningManager::prune(node, i);

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

void ActionSpace::expand_special_type(MiniNode *node, BaseNode *parent, int chosen_index, abc_t before, bool force_apply) const
{
    auto st = static_cast<SpecialType>(parent->get_action_at(chosen_index));
    const auto &aff = parent->get_affected_at(chosen_index);
    if ((st == SpecialType::CLL) || (st == SpecialType::CLR))
    {
        auto offset = (st == SpecialType::CLL) ? -1 : 1;
        auto char_map = map<abc_t, size_t>();
        for (const auto &item : aff)
        {
            auto order = item.first;
            auto pos = item.second;
            auto unit = node->base->words[order]->id_seq[pos + offset];
            auto base_unit = word_space->opt.unit2base[unit];
            // FIXME(j_luo) we really don't need to pass order and pos (and create item) separately -- we just need to decide whether to keep it or not.
            assert(cl_map.contains(base_unit));
            update_affected(node, cl_map.at(base_unit), order, pos, char_map, false);
        }
        // std::cerr << "CLLR\n";

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
    else
    {
        // Must use `unit2base` since stress is included for before.
        if (st == SpecialType::GBJ)
            ActionManager::add_action(node, gbj_map.at(before), aff);
        else if (st == SpecialType::GBW)
            ActionManager::add_action(node, gbw_map.at(before), aff);
        else if (force_apply)
            // HACK(j_luo) this is hacky.
            for (abc_t after_id = 0; after_id < 1000; ++after_id)
                ActionManager::add_action(node, after_id, aff);
        else
            for (abc_t after_id : permissible_changes.at(before))
                ActionManager::add_action(node, after_id, aff);
    }
}

void ActionSpace::expand_before(MiniNode *node, int chosen_index) const
{
    auto unit = node->base->get_action_at(chosen_index);
    const auto &aff = node->base->get_affected_at(chosen_index);
    ActionManager::add_action(node, static_cast<abc_t>(SpecialType::NONE), aff);
    if (word_space->opt.is_vowel[unit])
        ActionManager::add_action(node, static_cast<abc_t>(SpecialType::VS), aff);

    // CLL and CLR
    // std::cerr << "cllr\n";
    auto full_aff = node->get_affected_at(0);
    auto cll_aff = Affected();
    auto clr_aff = Affected();
    for (const auto &item : full_aff)
    {
        auto order = item.first;
        auto pos = item.second;
        auto &id_seq = node->base->words[order]->id_seq;
        if (pos > 0)
        {
            auto base_unit = word_space->opt.unit2base[id_seq[pos - 1]];
            if (cl_map.contains(base_unit))
                cll_aff.push_back(item);
        }
        if (pos < (id_seq.size() - 1))
        {
            auto base_unit = word_space->opt.unit2base[id_seq[pos + 1]];
            if (cl_map.contains(base_unit))
                clr_aff.push_back(item);
        }
    }
    if (cll_aff.size() > 0)
        ActionManager::add_action(node, static_cast<abc_t>(SpecialType::CLL), cll_aff);
    if (clr_aff.size() > 0)
        ActionManager::add_action(node, static_cast<abc_t>(SpecialType::CLR), clr_aff);
    // GBJ
    // std::cerr << "gbj\n";
    auto base_unit = word_space->opt.unit2base[unit];
    if (gbj_map.contains(base_unit))
        ActionManager::add_action(node, static_cast<abc_t>(SpecialType::GBJ), full_aff);
    // GBW
    // std::cerr << "gbw\n";
    if (gbw_map.contains(base_unit))
        ActionManager::add_action(node, static_cast<abc_t>(SpecialType::GBW), full_aff);
}

void ActionSpace::expand_normal(MiniNode *node, BaseNode *parent, int chosen_index, int offset, bool use_vowel_seq, bool can_have_null, bool can_have_any) const
{
    if (can_have_null)
        expand_null(node, parent, chosen_index);

    const auto &words = node->base->words;
    const auto &affected = parent->get_affected_at(chosen_index);
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
                update_affected(node, vowel_seq[vowel_pos], order, aff.second, char_map, can_have_any);
        }
        else
        {
            auto pos = aff.second + offset;
            auto &id_seq = words[order]->id_seq;
            if (in_bound(pos, id_seq.size()))
                update_affected(node, id_seq[pos], order, aff.second, char_map, can_have_any);
        }
    }
}

bool ActionSpace::expand_null_only(MiniNode *node, BaseNode *parent, int chosen_index) const
{
    abc_t last_unit = parent->get_action_at(chosen_index);
    if ((last_unit == opt.null_id) || (last_unit == opt.any_id) || (last_unit == opt.any_s_id) || (last_unit == opt.any_uns_id))
    {
        SPDLOG_TRACE("Phase {}, keeping only Null action.", str::from(node->ap));
        expand_null(node, parent, chosen_index);
        return true;
    }
    return false;
}

void ActionSpace::expand_after(MiniNode *node, BaseNode *parent, int chosen_index, bool use_vowel_seq, bool can_have_null, bool can_have_any) const { expand_normal(node, parent, chosen_index, -1, use_vowel_seq, can_have_null, can_have_any); }

void ActionSpace::expand_pre(MiniNode *node, BaseNode *parent, int chosen_index, bool use_vowel_seq, bool can_have_any) const
{
    if (!expand_null_only(node, parent, chosen_index))
        expand_normal(node, parent, chosen_index, -2, use_vowel_seq, true, can_have_any);
}

void ActionSpace::expand_d_pre(MiniNode *node, BaseNode *parent, int chosen_index, bool use_vowel_seq, bool can_have_null, bool can_have_any) const { expand_normal(node, parent, chosen_index, 1, use_vowel_seq, can_have_null, can_have_any); }

void ActionSpace::expand_post(MiniNode *node, BaseNode *parent, int chosen_index, bool use_vowel_seq, bool can_have_any) const
{
    if (!expand_null_only(node, parent, chosen_index))
        expand_normal(node, parent, chosen_index, 2, use_vowel_seq, true, can_have_any);
}

void ActionSpace::expand_null(MiniNode *node, BaseNode *parent, int chosen_index) const
{
    // Affected positions will not be further narrowed down.
    ActionManager::add_action(node, opt.null_id, parent->get_affected_at(chosen_index));
}

void ActionSpace::expand(MiniNode *node, const Subpath &subpath, bool use_vowel_seq, bool force_apply) const
{
    if (node->is_expanded())
    {
        SPDLOG_TRACE("MiniNode expanded already.");
        return;
    }

    if (node->stopped)
    {
        ActionManager::add_action(node, opt.null_id, Affected());
        SPDLOG_TRACE("Phase {}, keeping only Null action due to stopped status.", str::from(node->ap));
    }
    else
    {
        switch (node->ap)
        {
        case ActionPhase::BEFORE:
            expand_before(node, subpath.chosen_seq[0].first);
            break;
        case ActionPhase::SPECIAL_TYPE:
        {
            abc_t before = word_space->opt.unit2base[subpath.chosen_seq[0].second];
            expand_special_type(node, subpath.mini_node_seq[0], subpath.chosen_seq[1].first, before, force_apply);
            break;
        }
        case ActionPhase::AFTER:
        {
            bool can_have_null = (static_cast<SpecialType>(subpath.chosen_seq[1].second) != SpecialType::CLL);
            expand_after(node, subpath.mini_node_seq[1], subpath.chosen_seq[2].first, use_vowel_seq, can_have_null, can_have_null);
            break;
        }
        case ActionPhase::PRE:
            expand_pre(node, subpath.mini_node_seq[2], subpath.chosen_seq[3].first, use_vowel_seq, true);
            break;
        case ActionPhase::D_PRE:
        {
            bool can_have_null = (static_cast<SpecialType>(subpath.chosen_seq[1].second) != SpecialType::CLR);
            expand_d_pre(node, subpath.mini_node_seq[3], subpath.chosen_seq[4].first, use_vowel_seq, can_have_null, can_have_null);
            break;
        }
        case ActionPhase::POST:
            expand_post(node, subpath.mini_node_seq[4], subpath.chosen_seq[5].first, use_vowel_seq, true);
            break;
        }
    }

    expand_stats(node);
    SPDLOG_DEBUG("ActionSpace:: mini node expanded with #actions {}.", node->get_num_actions());

    if (!node->stopped)
        for (size_t i = 0; i < node->get_num_actions(); ++i)
            if (node->get_num_affected_at(i) < opt.site_threshold)
                PruningManager::prune(node, i);

    // std::cerr << "-----------------\n";
    // std::cerr << str::from(node->ap) << " " << node->permissible_chars.size() << "\n";

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
    return;
}

void ActionSpace::update_affected(BaseNode *node, abc_t unit, int order, size_t pos, map<abc_t, size_t> &char_map, bool can_have_any) const
{
    // FIXME(j_luo) clearer logic here -- e.g., <any> is not aviable for after_id.
    if (((unit == opt.any_id) || (unit == opt.any_s_id) || (unit == opt.any_uns_id)) && !can_have_any)
        return;

    if (!char_map.contains(unit))
    {
        // Add one more permission char.
        char_map[unit] = node->get_num_actions();
        ActionManager::add_action(node, unit, Affected{{{order, pos}}});
    }
    else
    {
        // Add one more position.
        ActionManager::update_affected_at(node, char_map[unit], order, pos);
    }

    Stress stress = word_space->opt.unit_stress[unit];
    if (stress != Stress::NOSTRESS)
    {
        update_affected(node, word_space->opt.unit2base[unit], order, pos, char_map, can_have_any);
        if (stress == Stress::STRESSED)
        {
            if (unit != opt.any_s_id)
                update_affected(node, opt.any_s_id, order, pos, char_map, can_have_any);
        }
        else if (unit != opt.any_uns_id)
            update_affected(node, opt.any_uns_id, order, pos, char_map, can_have_any);
    }
    else if ((unit != opt.any_id) && (!word_space->opt.is_vowel[unit]))
        update_affected(node, opt.any_id, order, pos, char_map, can_have_any);
}

void ActionSpace::evaluate(MiniNode *node) const
{
    ActionManager::evaluate(node);
}

void ActionSpace::expand_stats(BaseNode *node) const
{
    size_t n = node->get_num_actions();
    clear_stats(node, false);
    clear_priors(node, false);
    EdgeBuilder::init_edges(node);
    ActionManager::init_pruned(node);
    if (node->is_transitional())
        ActionManager::init_rewards(static_cast<TransitionNode *>(node));
}

void ActionSpace::clear_stats(BaseNode *root, bool recursive) const
{
    auto queue = recursive ? Traverser::bfs(root) : vec<BaseNode *>{root};
    for (const auto node : queue)
        ActionManager::init_stats(node);
}

void ActionSpace::clear_priors(BaseNode *root, bool recursive) const
{
    auto queue = recursive ? Traverser::bfs(root) : vec<BaseNode *>{root};
    for (const auto node : queue)
        ActionManager::clear_priors(node);
}

void ActionSpace::evaluate(TreeNode *node, const vec<vec<float>> &meta_priors, const vec<float> &special_priors) { ActionManager::evaluate(node, meta_priors, special_priors); }

void ActionSpace::add_noise(TreeNode *node, const vec<vec<float>> &meta_noise, const vec<float> &special_noise, float noise_ratio) const
{
    ActionManager::add_noise(node, meta_noise, special_noise, noise_ratio);
}

void ActionSpace::connect(BaseNode *base, const Subpath &subpath) const
{
    BaseNode *parent = base;
    for (size_t i = 0; i < 6; ++i)
    {
        auto child = subpath.mini_node_seq[i];
        auto index = subpath.chosen_seq[i].first;
        EdgeBuilder::connect(parent, index, child);
        parent = child;
    }
}
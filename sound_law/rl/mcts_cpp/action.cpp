#include "action.hpp"

void Subpath::connect(TreeNode *node) const
{
    BaseNode *parent = node;
    for (size_t i = 0; i < 6; ++i)
    {
        auto child = mini_node_seq[i];
        auto index = chosen_seq[i].first;
        if (parent->children[index] == nullptr)
            parent->children[index] = child;
        parent = child;
    }
}

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
        new_node = new TreeNode(node->words, node->depth + 1, last, subpath.chosen_seq[6], true);
        EdgeBuilder::connect(last, last_child_index, new_node);
    }
    // new_node = TreeNode::get_tree_node(node->words, node->depth + 1, last, subpath.chosen_seq[6], true);
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
            auto new_id_seq = change_id_seq(node->words[order]->id_seq, item.second, after_id, st);
            auto new_word = word_space->get_word(new_id_seq);
            new_words[order] = new_word;
            word_space->set_edit_dist_at(new_word, order);
        }
        // new_node = new TreeNode(new_words, node->depth + 1, last, subpath.chosen_seq[6], false);
        new_node = TreeNode::get_tree_node(new_words, node->depth + 1, last, subpath.chosen_seq[6], false);
        EdgeBuilder::connect(last, last_child_index, new_node);
        expand(new_node);
        if ((node->dist - new_node->dist) < opt.dist_threshold)
            last->prune(last_child_index);
        // new_node->prune();
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
    Subpath subpath = Subpath();

    // std::cerr << "before\n";
    auto before = ChosenChar({get_index(before_id, node->permissible_chars), before_id});
    bool stopped = (before.first == 0);
    auto before_mn = get_mini_node(node, node, before, ActionPhase::BEFORE, stopped);
    subpath.chosen_seq[0] = before;
    subpath.mini_node_seq[0] = before_mn;
    subpath.stopped = stopped;
    expand(before_mn, subpath, false, false);
    dummy_evaluate(before_mn->priors, before_mn->permissible_chars.size());

    // std::cerr << "special_type\n";
    abc_t st_abc = static_cast<abc_t>(st);
    auto special_type = ChosenChar({get_index(st_abc, before_mn->permissible_chars), st_abc});
    auto st_mn = get_mini_node(node, before_mn, special_type, ActionPhase::SPECIAL_TYPE, stopped);
    subpath.chosen_seq[1] = special_type;
    subpath.mini_node_seq[1] = st_mn;
    expand(st_mn, subpath, false, true);
    dummy_evaluate(st_mn->priors, st_mn->permissible_chars.size());

    // std::cerr << "after\n";
    // std::cerr << after_id << "\n";
    // for (const auto x : st_mn->permissible_chars)
    //     std::cerr << x << " ";
    // std::cerr << "\n";
    auto after = ChosenChar({get_index(after_id, st_mn->permissible_chars), after_id});
    auto after_mn = get_mini_node(node, st_mn, after, ActionPhase::AFTER, stopped);
    bool use_vowel_seq = (st == SpecialType::VS);
    subpath.chosen_seq[2] = after;
    subpath.mini_node_seq[2] = after_mn;
    expand(after_mn, subpath, use_vowel_seq, false);
    dummy_evaluate(after_mn->priors, after_mn->permissible_chars.size());

    // std::cerr << "pre\n";
    auto pre = ChosenChar({get_index(pre_id, after_mn->permissible_chars), pre_id});
    auto pre_mn = get_mini_node(node, after_mn, pre, ActionPhase::PRE, stopped);
    subpath.chosen_seq[3] = pre;
    subpath.mini_node_seq[3] = pre_mn;
    expand(pre_mn, subpath, use_vowel_seq, false);
    dummy_evaluate(pre_mn->priors, pre_mn->permissible_chars.size());

    auto d_pre = ChosenChar({get_index(d_pre_id, pre_mn->permissible_chars), d_pre_id});
    // std::cerr << "d_pre\n";
    auto d_pre_mn = get_mini_node(node, pre_mn, d_pre, ActionPhase::D_PRE, stopped);
    subpath.chosen_seq[4] = d_pre;
    subpath.mini_node_seq[4] = d_pre_mn;
    expand(d_pre_mn, subpath, use_vowel_seq, false);
    dummy_evaluate(d_pre_mn->priors, d_pre_mn->permissible_chars.size());

    // std::cerr << "post\n";
    auto post = ChosenChar({get_index(post_id, d_pre_mn->permissible_chars), post_id});
    auto post_mn = get_mini_node(node, d_pre_mn, post, ActionPhase::POST, stopped);
    subpath.chosen_seq[5] = post;
    subpath.mini_node_seq[5] = post_mn;
    expand(post_mn, subpath, use_vowel_seq, false);
    dummy_evaluate(post_mn->priors, post_mn->permissible_chars.size());

    // std::cerr << "d_post\n";
    auto d_post = ChosenChar({get_index(d_post_id, post_mn->permissible_chars), d_post_id});
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

Subpath ActionSpace::get_best_subpath(TreeNode *node, float puct_c, int game_count, float virtual_loss, float heur_c)
{
    Subpath subpath = Subpath();

    // std::cerr << "before\n";
    SPDLOG_DEBUG("ActionSpace:: getting best subpath...");
    auto before = node->get_best_subaction(puct_c, game_count, virtual_loss, heur_c);
    bool stopped = (before.first == 0);
    auto before_mn = get_mini_node(node, node, before, ActionPhase::BEFORE, stopped);
    subpath.chosen_seq[0] = before;
    subpath.mini_node_seq[0] = before_mn;
    subpath.stopped = stopped;
    expand(before_mn, subpath, false, false);
    evaluate(before_mn);
    SPDLOG_DEBUG("ActionSpace:: before done.");

    // std::cerr << "special_type\n";
    auto special_type = before_mn->get_best_subaction(puct_c, game_count, virtual_loss, heur_c);
    auto st_mn = get_mini_node(node, before_mn, special_type, ActionPhase::SPECIAL_TYPE, stopped);
    subpath.chosen_seq[1] = special_type;
    subpath.mini_node_seq[1] = st_mn;
    expand(st_mn, subpath, false, false);
    evaluate(st_mn);
    SPDLOG_DEBUG("ActionSpace:: special_type done.");

    // std::cerr << "after\n";
    auto after = st_mn->get_best_subaction(puct_c, game_count, virtual_loss, heur_c);
    bool use_vowel_seq = (static_cast<SpecialType>(special_type.second) == SpecialType::VS);
    auto after_mn = get_mini_node(node, st_mn, after, ActionPhase::AFTER, stopped);
    subpath.chosen_seq[2] = after;
    subpath.mini_node_seq[2] = after_mn;
    expand(after_mn, subpath, use_vowel_seq, false);
    evaluate(after_mn);
    SPDLOG_DEBUG("ActionSpace:: after done.");

    // std::cerr << "pre\n";
    auto pre = after_mn->get_best_subaction(puct_c, game_count, virtual_loss, heur_c);
    auto pre_mn = get_mini_node(node, after_mn, pre, ActionPhase::PRE, stopped);
    subpath.chosen_seq[3] = pre;
    subpath.mini_node_seq[3] = pre_mn;
    expand(pre_mn, subpath, use_vowel_seq, false);
    evaluate(pre_mn);
    SPDLOG_DEBUG("ActionSpace:: pre done.");

    // std::cerr << "d_pre\n";
    auto d_pre = pre_mn->get_best_subaction(puct_c, game_count, virtual_loss, heur_c);
    auto d_pre_mn = get_mini_node(node, pre_mn, d_pre, ActionPhase::D_PRE, stopped);
    subpath.chosen_seq[4] = d_pre;
    subpath.mini_node_seq[4] = d_pre_mn;
    expand(d_pre_mn, subpath, use_vowel_seq, false);
    evaluate(d_pre_mn);
    SPDLOG_DEBUG("ActionSpace:: d_pre done.");

    // std::cerr << "post\n";
    auto post = d_pre_mn->get_best_subaction(puct_c, game_count, virtual_loss, heur_c);
    auto post_mn = get_mini_node(node, d_pre_mn, post, ActionPhase::POST, stopped);
    subpath.chosen_seq[5] = post;
    subpath.mini_node_seq[5] = post_mn;
    expand(post_mn, subpath, use_vowel_seq, false);
    evaluate(post_mn);
    SPDLOG_DEBUG("ActionSpace:: post done.");

    // std::cerr << "d_post\n";
    auto d_post = post_mn->get_best_subaction(puct_c, game_count, virtual_loss, heur_c);
    subpath.chosen_seq[6] = d_post;
    SPDLOG_DEBUG("ActionSpace:: d_post done.");

    subpath.connect(node);
    return subpath;
}

MiniNode *ActionSpace::get_mini_node(TreeNode *base, BaseNode *parent, const ChosenChar &chosen, ActionPhase ap, bool stopped)
{
    // BaseNode *&child = parent->children[chosen.first];
    bool is_transition = (ap == ActionPhase::POST);
    BaseNode *child;
    if (!parent->has_child(chosen.first))
    {
        if (is_transition)
            child = new TransitionNode(base, static_cast<MiniNode *>(parent), chosen, stopped);
        else
            child = new MiniNode(base, parent, chosen, ap, stopped);
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
                update_affected(node, id_seq[pos], order, pos, char_map, false);
    }

    expand_stats(node);
    SPDLOG_DEBUG("ActionSpace:: node expanded with #actions {}.", node->permissible_chars.size());

    // Skip STOP.
    for (size_t i = 1; i < node->permissible_chars.size(); ++i)
        if (node->affected[i].size() < opt.site_threshold)
            node->prune(i);

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

void ActionSpace::expand_special_type(MiniNode *node, BaseNode *parent, int chosen_index, abc_t before, bool force_apply)
{
    auto st = static_cast<SpecialType>(parent->permissible_chars[chosen_index]);
    if ((st == SpecialType::CLL) || (st == SpecialType::CLR))
    {
        const auto &aff = parent->affected[chosen_index];
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
            update_affected(node, cl_map[base_unit], order, pos, char_map, false);
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
            node->permissible_chars.push_back(gbj_map[before]);
        else if (st == SpecialType::GBW)
            node->permissible_chars.push_back(gbw_map[before]);
        else if (force_apply)
            // HACK(j_luo) this is hacky.
            for (abc_t after_id = 0; after_id < 1000; ++after_id)
                node->permissible_chars.push_back(after_id);
        else
            node->permissible_chars = permissible_changes[before];
        // FIXME(j_luo) This is not very efficient.
        // std::cerr << "node\n";
        // std::cerr << node->parent->chosen_char.second << "\n";
        node->affected = vec<Affected>(node->permissible_chars.size(), parent->affected[chosen_index]);
    }
}
void ActionSpace::expand_before(MiniNode *node, BaseNode *parent, int chosen_index)
{
    auto unit = parent->permissible_chars[chosen_index];
    node->permissible_chars.push_back(static_cast<abc_t>(SpecialType::NONE));
    if (word_space->opt.is_vowel[unit])
        node->permissible_chars.push_back(static_cast<abc_t>(SpecialType::VS));
    node->affected = vec<Affected>(node->permissible_chars.size(), parent->affected[chosen_index]);

    // CLL and CLR
    // std::cerr << "cllr\n";
    auto cll_aff = Affected();
    auto clr_aff = Affected();
    for (const auto &item : node->affected[0])
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
    {
        node->permissible_chars.push_back(static_cast<abc_t>(SpecialType::CLL));
        node->affected.push_back(cll_aff);
    }
    if (clr_aff.size() > 0)
    {
        node->permissible_chars.push_back(static_cast<abc_t>(SpecialType::CLR));
        node->affected.push_back(clr_aff);
    }
    // GBJ
    // std::cerr << "gbj\n";
    auto base_unit = word_space->opt.unit2base[unit];
    if (gbj_map.contains(base_unit))
    {
        node->permissible_chars.push_back(static_cast<abc_t>(SpecialType::GBJ));
        node->affected.push_back(node->affected[0]);
    }
    // GBW
    // std::cerr << "gbw\n";
    if (gbw_map.contains(base_unit))
    {
        node->permissible_chars.push_back(static_cast<abc_t>(SpecialType::GBW));
        node->affected.push_back(node->affected[0]);
    }
}
void ActionSpace::expand_normal(MiniNode *node, BaseNode *parent, int chosen_index, int offset, bool use_vowel_seq, bool can_have_null, bool can_have_any)
{
    if (can_have_null)
        expand_null(node, parent, chosen_index);

    const auto &words = node->base->words;
    const auto &affected = parent->affected[chosen_index];
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
bool ActionSpace::expand_null_only(MiniNode *node, BaseNode *parent, int chosen_index)
{
    abc_t last_unit = parent->permissible_chars[chosen_index];
    if ((last_unit == opt.null_id) || (last_unit == opt.any_id) || (last_unit == opt.any_s_id) || (last_unit == opt.any_uns_id))
    {
        SPDLOG_TRACE("Phase {}, keeping only Null action.", str::from(node->ap));
        expand_null(node, parent, chosen_index);
        return true;
    }
    return false;
}
void ActionSpace::expand_after(MiniNode *node, BaseNode *parent, int chosen_index, bool use_vowel_seq, bool can_have_null, bool can_have_any) { expand_normal(node, parent, chosen_index, -1, use_vowel_seq, can_have_null, can_have_any); }
void ActionSpace::expand_pre(MiniNode *node, BaseNode *parent, int chosen_index, bool use_vowel_seq, bool can_have_any)
{
    if (!expand_null_only(node, parent, chosen_index))
        expand_normal(node, parent, chosen_index, -2, use_vowel_seq, true, can_have_any);
}
void ActionSpace::expand_d_pre(MiniNode *node, BaseNode *parent, int chosen_index, bool use_vowel_seq, bool can_have_null, bool can_have_any) { expand_normal(node, parent, chosen_index, 1, use_vowel_seq, can_have_null, can_have_any); }
void ActionSpace::expand_post(MiniNode *node, BaseNode *parent, int chosen_index, bool use_vowel_seq, bool can_have_any)
{
    if (!expand_null_only(node, parent, chosen_index))
        expand_normal(node, parent, chosen_index, 2, use_vowel_seq, true, can_have_any);
}
void ActionSpace::expand_null(MiniNode *node, BaseNode *parent, int chosen_index)
{
    node->permissible_chars.push_back(opt.null_id);
    // Affected positions will not be further narrowed down.
    node->affected = vec<Affected>({parent->affected[chosen_index]});
}

void ActionSpace::expand(MiniNode *node, const Subpath &subpath, bool use_vowel_seq, bool force_apply)
{
    std::lock_guard<std::mutex> lock(node->mtx);

    if (node->is_expanded())
    {
        SPDLOG_TRACE("MiniNode expanded already.");
        return;
    }

    // SPDLOG_TRACE("ActionSpace:: Expanding {0}, chosen_char: ({1}, {2}), parent #actions {3}",
    //              str::from(node->ap), node->chosen_char.first, node->chosen_char.second, node->parent->permissible_chars.size());

    if (node->stopped)
    {
        node->permissible_chars.push_back(opt.null_id);
        node->affected = vec<Affected>({{}});
        SPDLOG_TRACE("Phase {}, keeping only Null action due to stopped status.", str::from(node->ap));
    }
    else
    {
        switch (node->ap)
        {
        case ActionPhase::BEFORE:
            expand_before(node, node->base, subpath.chosen_seq[0].first);
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
    SPDLOG_DEBUG("ActionSpace:: mini node expanded with #actions {}.", node->permissible_chars.size());

    if (!node->stopped)
        for (size_t i = 0; i < node->permissible_chars.size(); ++i)
            if (node->affected[i].size() < opt.site_threshold)
                node->prune(i);

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

void ActionSpace::update_affected(BaseNode *node, abc_t unit, int order, size_t pos, map<abc_t, size_t> &char_map, bool can_have_any)
{
    // FIXME(j_luo) clearer logic here -- e.g., <any> is not aviable for after_id.
    if (((unit == opt.any_id) || (unit == opt.any_s_id) || (unit == opt.any_uns_id)) && !can_have_any)
        return;

    if (!char_map.contains(unit))
    {
        // Add one more permission char.
        char_map[unit] = node->permissible_chars.size();
        node->permissible_chars.push_back(unit);
        node->affected.push_back(Affected{{{order, pos}}});
    }
    else
    {
        // Add one more position.
        auto &aff = node->affected[char_map[unit]];
        aff.push_back({order, pos});
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

inline void normalize(vec<float> &priors)
{
    float sum = 1e-8;
    for (const auto prior : priors)
        sum += prior;
    for (auto &prior : priors)
        prior /= sum;
}

void ActionSpace::evaluate(MiniNode *node)
{
    assert(node->is_expanded());
    if (node->is_evaluated())
        return;
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
    normalize(node->priors);
}

void ActionSpace::expand_stats(BaseNode *node)
{
    size_t n = node->permissible_chars.size();
    clear_stats(node, false);
    clear_priors(node, false);
    node->num_unpruned_actions = n;
    node->children = vec<BaseNode *>(n, nullptr);
    node->pruned = vec<bool>(n, false);
    if (node->is_transitional())
        static_cast<TransitionNode *>(node)->rewards = vec<float>(n, 0.0);
}

void ActionSpace::clear_stats(BaseNode *root, bool recursive)
{
    auto queue = recursive ? Traverser::bfs(root) : vec<BaseNode *>{root};
    for (const auto node : queue)
    {
        size_t n = node->permissible_chars.size();
        node->action_counts = vec<visit_t>(n, 0);
        node->total_values = vec<float>(n, 0.0);
        node->visit_count = 0;
        node->max_index = -1;
        node->max_value = -9999.9;
        node->max_values = vec<float>(n, -9999.9);
        // node->played = false;
    }
}

void ActionSpace::clear_priors(BaseNode *root, bool recursive)
{
    auto queue = recursive ? Traverser::bfs(root) : vec<BaseNode *>{root};
    for (const auto node : queue)
        node->priors.clear();
}

void ActionSpace::prune(BaseNode *node, bool include_self)
{
    for (const auto child : node->children)
        if (child != nullptr)
        {
            prune(child, false);
            delete child;
        }
    if (include_self)
    {
        assert(false);
        // EdgeBuilder::disconnect_from_parents(node);
        delete node;
    }
    else
        std::fill(node->children.begin(), node->children.end(), nullptr);
}

void ActionSpace::evaluate(TreeNode *node, const vec<vec<float>> &meta_priors, const vec<float> &special_priors)
{
    assert(node->is_expanded());
    if (node->is_evaluated())
        return;
    node->meta_priors = meta_priors;
    node->special_priors = special_priors;
    node->priors.clear();
    node->priors.reserve(node->permissible_chars.size());
    auto &full_priors = node->meta_priors[0];
    for (const auto unit : node->permissible_chars)
        // {
        //     std::cerr << full_priors.size() << " " << unit << "\n";
        node->priors.push_back(full_priors[unit]);
    // }
    normalize(node->priors);
}

void ActionSpace::add_noise(TreeNode *node, const vec<vec<float>> &meta_noise, const vec<float> &special_noise, float noise_ratio)
{
    auto new_meta_priors = node->meta_priors;
    auto new_special_priors = node->special_priors;
    for (size_t i = 0; i < meta_noise.size(); ++i)
        for (size_t j = 0; j < meta_noise[i].size(); ++j)
            new_meta_priors[i][j] = new_meta_priors[i][j] * (1.0 - noise_ratio) + meta_noise[i][j] * noise_ratio;
    for (size_t i = 0; i < special_noise.size(); ++i)
        new_special_priors[i] = new_special_priors[i] * (1.0 - noise_ratio) + special_noise[i] * noise_ratio;
    evaluate(node, new_meta_priors, new_special_priors);
}
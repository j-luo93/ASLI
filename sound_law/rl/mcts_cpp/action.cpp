#include "action.hpp"
#include "timer.hpp"
#include "stats.hpp"

ActionSpace::ActionSpace(
    SiteSpace *site_space,
    WordSpace *word_space,
    float dist_threshold,
    int site_threshold) : site_space(site_space),
                          word_space(word_space),
                          dist_threshold(dist_threshold),
                          site_threshold(site_threshold) {}

void ActionSpace::register_edge(abc_t before_id, abc_t after_id)
{
    edges[before_id].push_back(std::make_pair(SpecialType::NONE, after_id));
}

void ActionSpace::register_cl_map(abc_t before_id, abc_t after_id)
{
    edges[before_id].push_back(std::make_pair(SpecialType::CLL, after_id));
    edges[before_id].push_back(std::make_pair(SpecialType::CLR, after_id));
}

void ActionSpace::register_gbj(abc_t before_id, abc_t after_id) { edges[before_id].push_back(std::make_pair(SpecialType::GBJ, after_id)); }
void ActionSpace::register_gbw(abc_t before_id, abc_t after_id) { edges[before_id].push_back(std::make_pair(SpecialType::GBW, after_id)); }

void ActionSpace::set_vowel_info(const vec<bool> &vowel_mask,
                                 const vec<abc_t> &vowel_base,
                                 const vec<Stress> &vowel_stress,
                                 const vec<abc_t> &stressed_vowel,
                                 const vec<abc_t> &unstressed_vowel)
{
    this->vowel_mask = vowel_mask;
    this->vowel_base = vowel_base;
    this->vowel_stress = vowel_stress;
    this->stressed_vowel = stressed_vowel;
    this->unstressed_vowel = unstressed_vowel;
    site_space->set_vowel_info(vowel_mask, vowel_base, vowel_stress);
}

void ActionSpace::set_glide_info(abc_t glide_j, abc_t glide_w)
{
    this->glide_j = glide_j;
    this->glide_w = glide_w;
}

void ActionSpace::set_action_allowed(Pool *tp, const vec<TreeNode *> &tnodes)
{
    // Find unique tree nodes first.
    SPDLOG_DEBUG("Finding unique tree nodes.");
    auto unique_tnodes = vec<TreeNode *>();
    find_unique(unique_tnodes, tnodes, [](auto &&input) { return true; });

    parallel_apply<1>(
        tp,
        [this](TreeNode *tnode) { set_action_allowed(tnode); },
        unique_tnodes);
}

void ActionSpace::add_actions_from_graph(TreeNode *tnode, const SiteGraph &graph, vec<uai_t> &actions, bool use_vowel_seq)
{
    for (const auto &item : graph.nodes)
    {
        GraphNode *gnode = item.second;
        if (gnode->num_sites < site_threshold)
            continue;

        // Do not include unconditional changes if we are using vowel sequence -- they would have been covered by normal sequences anyway.
        if (use_vowel_seq && (gnode->children.size() == 0))
            continue;

        bool can_be_simpler = false;
        for (auto child : gnode->children)
            if (child->num_sites == gnode->num_sites)
            {
                can_be_simpler = true;
                break;
            }
        if (can_be_simpler)
            continue;

        usi_t site = gnode->base->site;
        abc_t before_id = site::get_before_id(site);
        auto &st_and_after_ids = edges[before_id];
        size_t orig_n = st_and_after_ids.size();
        auto action_ids = vec<uai_t>();
        action_ids.reserve(orig_n);
        for (auto &st_and_after : st_and_after_ids)
        {
            SpecialType st = st_and_after.first;
            if (use_vowel_seq && (st != SpecialType::NONE))
                break;
            uai_t action_id;
            if (use_vowel_seq)
                action_id = action::combine_after_id_special(site, st_and_after.second, SpecialType::VS);
            else
                action_id = action::combine_after_id_special(site, st_and_after.second, st);
            action_ids.push_back(action_id);
        }
        size_t n = action_ids.size();
        auto deltas = vec<float>(n);
        for (auto order : gnode->linked_words)
        {
            auto word = tnode->words[order];
            bool is_short = (word->id_seq.size() == 3);
            auto new_words = vec<Word *>();
            apply_actions(new_words, word, site, action_ids, use_vowel_seq);
            for (size_t i = 0; i < n; i++)
            {
                abc_t after_id = st_and_after_ids[i].second;
                if ((after_id == site_space->emp_id) && is_short)
                {
                    deltas[i] += 999999999.9;
                    continue;
                }
                auto new_word = new_words[i];
                SPDLOG_TRACE("  new word {0} old word {1} order {2}", new_word->str(), word->str(), order);
                deltas[i] += word_space->safe_get_dist(new_word, order) - word_space->safe_get_dist(word, order);
            }
        }
        for (size_t i = 0; i < n; i++)
            if (deltas[i] < dist_threshold)
                actions.push_back(action_ids[i]);
    }
}

void ActionSpace::set_action_allowed(TreeNode *tnode)
{
    if (!tnode->action_allowed.empty() || (tnode->done) || (tnode->stopped))
        return;

    // Build the graph first.
    SiteGraph graph = SiteGraph();
    SiteGraph vgraph = SiteGraph();
    SPDLOG_TRACE("Getting graph outputs.");
    for (size_t order = 0; order < tnode->words.size(); order++)
    {
        Word *word = tnode->words[order];
        for (SiteNode *root : word->site_roots)
            graph.add_root(root, order);
        for (SiteNode *vroot : word->vowel_site_roots)
            vgraph.add_root(vroot, order);
    }

    auto &aa = tnode->action_allowed;
    aa.reserve(1000);
    // Stop is always available.
    aa.push_back(action::STOP);
    add_actions_from_graph(tnode, graph, aa, false);
    add_actions_from_graph(tnode, vgraph, aa, true);
}

int ActionSpace::locate_edge_index(abc_t before_id, SpecialType st, abc_t after_id, bool use_vowel_seq)
{
    auto &st_and_after_ids = edges[before_id];
    for (size_t i = 0; i < st_and_after_ids.size(); i++)
    {
        auto &edge = st_and_after_ids[i];
        if ((use_vowel_seq || (edge.first == st)) && (edge.second == after_id))
            return i;
    }
    return -1;
}

inline bool ActionSpace::match(abc_t idx, abc_t target)
{
    Stress tgt_stress = vowel_stress[target];
    if ((tgt_stress == Stress::STRESSED) || (tgt_stress == Stress::UNSTRESSED))
    {
        if (tgt_stress != vowel_stress[idx])
            return false;
    }

    if ((target == site_space->any_id) || (target == site_space->any_s_id) || (target == site_space->any_uns_id))
        return ((idx != site_space->sot_id) && (idx != site_space->eot_id) && (idx != site_space->syl_eot_id));

    return (vowel_base[target] == vowel_base[idx]);
}

inline IdSeq ActionSpace::apply_action(const IdSeq &id_seq, uai_t action_id)
{
    IdSeq new_id_seq = vec<abc_t>();
    abc_t before_id = action::get_before_id(action_id);
    abc_t after_id = action::get_after_id(action_id);
    abc_t pre_id = action::get_pre_id(action_id);
    abc_t d_pre_id = action::get_d_pre_id(action_id);
    abc_t post_id = action::get_post_id(action_id);
    abc_t d_post_id = action::get_d_post_id(action_id);

    switch (vowel_stress[before_id])
    {
    case Stress::STRESSED:
        after_id = stressed_vowel[after_id];
        break;
    case Stress::UNSTRESSED:
        after_id = unstressed_vowel[after_id];
        break;
    default:
        break;
    }

    bool syncope = (after_id == site_space->emp_id);
    SpecialType st = action::get_special_type(action_id);
    int n = id_seq.size();
    auto vowel_seq = IdSeq();
    auto orig_idx = vec<int>();
    bool use_vowel_seq = (st == SpecialType::VS);
    new_id_seq.push_back(site_space->sot_id);
    if (use_vowel_seq)
    {
        vowel_seq.reserve(id_seq.size());
        orig_idx.reserve(id_seq.size());
        vowel_seq.push_back(site_space->sot_id);
        for (int i = 1; i < n - 1; i++)
            if (vowel_mask[id_seq[i]])
            {
                vowel_seq.push_back(id_seq[i]);
                orig_idx.push_back(i);
            }
        vowel_seq.push_back((vowel_mask[id_seq[n - 2]]) ? site_space->syl_eot_id : site_space->eot_id);
        int m = vowel_seq.size();
        int j = 1;
        for (int i = 1; i < m - 1; i++)
        {
            bool applied = match(vowel_seq[i], before_id);
            if (applied && (pre_id != NULL_ABC))
            {
                if ((i < 1) || (!match(vowel_seq[i - 1], pre_id)))
                    applied = false;
                if (applied && (d_pre_id != NULL_ABC))
                    if ((i < 2) || (!match(vowel_seq[i - 2], d_pre_id)))
                        applied = false;
            }
            if (applied && (post_id != NULL_ABC))
            {
                if ((i > n - 2) || (!match(vowel_seq[i + 1], post_id)))
                    applied = false;
                if (applied && (d_post_id != NULL_ABC))
                    if ((i > n - 3) || (!match(vowel_seq[i + 2], d_post_id)))
                        applied = false;
            }

            int upper = orig_idx[i - 1];
            while (j < upper)
            {
                new_id_seq.push_back(id_seq[j]);
                j++;
            }
            if (applied)
                if (syncope)
                    continue;
                else
                    new_id_seq.push_back(after_id);
            else
                new_id_seq.push_back(vowel_seq[i]);
            j++;
        }
        while (j < n - 1)
        {
            new_id_seq.push_back(id_seq[j]);
            j++;
        }
    }
    else
    {
        for (int i = 1; i < n - 1; i++)
        {
            bool applied = match(id_seq[i], before_id);
            if (applied && (pre_id != NULL_ABC))
            {
                if ((i < 1) || (!match(id_seq[i - 1], pre_id)))
                    applied = false;
                if (applied && (d_pre_id != NULL_ABC))
                    if ((i < 2) || (!match(id_seq[i - 2], d_pre_id)))
                        applied = false;
            }
            if (applied && (post_id != NULL_ABC))
            {
                if ((i > n - 2) || (!match(id_seq[i + 1], post_id)))
                    applied = false;
                if (applied && (d_post_id != NULL_ABC))
                    if ((i > n - 3) || (!match(id_seq[i + 2], d_post_id)))
                        applied = false;
            }
            if (applied)
                if (syncope)
                    continue;
                else
                {
                    switch (st)
                    {
                    case SpecialType::NONE:
                        new_id_seq.push_back(after_id);
                        break;
                    case SpecialType::CLL:
                        new_id_seq.pop_back();
                        new_id_seq.push_back(after_id);
                        break;
                    case SpecialType::CLR:
                        new_id_seq.push_back(after_id);
                        i++;
                        break;
                    case SpecialType::GBJ:
                        new_id_seq.push_back(glide_j);
                        new_id_seq.push_back(after_id);
                        break;
                    case SpecialType::GBW:
                        new_id_seq.push_back(glide_w);
                        new_id_seq.push_back(after_id);
                        break;
                    }
                }
            else
                new_id_seq.push_back(id_seq[i]);
        }
    }
    new_id_seq.push_back(site_space->eot_id);
    return new_id_seq;
}

void ActionSpace::apply_actions(vec<Word *> &outputs, Word *word, usi_t site, const vec<uai_t> &action_ids, bool use_vowel_seq)
{
    auto &neighbors = (use_vowel_seq) ? word->vowel_neighbors : word->neighbors;
    if (neighbors.if_contains(site, [&outputs](vec<Word *> const &values) { outputs = values; }))
        return;

    outputs.reserve(action_ids.size());
    for (uai_t action_id : action_ids)
    {
        auto new_id_seq = apply_action(word->id_seq, action_id);
        outputs.push_back(nullptr);
        word_space->get_word(outputs.back(), new_id_seq);
    }
    // NOTE(j_luo) No need to do anything if the key exists -- `get_word` ensures the right Word object is returned.
    neighbors.try_emplace_l(
        site, [](vec<Word *> &values) {}, outputs);
}

vec<uai_t> ActionSpace::get_similar_actions(uai_t action)
{
    auto site = action::get_site(action);
    auto after_id = action::get_after_id(action);

    SiteNode *site_node;
    site_space->get_node(site_node, site);
    auto graph = SiteGraph();
    auto root = graph.add_root(site_node, -1); // `order` doesn't matter here.
    auto desc = graph.get_descendants(root);

    auto ret = vec<uai_t>();
    for (const auto node : desc)
        ret.push_back(action::combine_after_id(node->base->site, after_id));
    return ret;
}
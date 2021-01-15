#include "action.hpp"

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
    edges[before_id].push_back(after_id);
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

void ActionSpace::set_action_allowed(TreeNode *tnode)
{
    if (!tnode->action_allowed.empty() || (tnode->done) || (tnode->stopped))
        return;

    // Build the graph first.
    SiteGraph graph = SiteGraph();
    SPDLOG_TRACE("Getting graph outputs.");
    for (size_t order = 0; order < tnode->words.size(); order++)
    {
        Word *word = tnode->words[order];
        for (SiteNode *root : word->site_roots)
            graph.add_root(root, order);
    }

    auto &aa = tnode->action_allowed;
    aa.reserve(1000);
    // Stop is always available.
    aa.push_back(action::STOP);
    for (const auto &item : graph.nodes)
    {
        GraphNode *gnode = item.second;
        if (gnode->num_sites < site_threshold)
            continue;
        if ((gnode->lchild != nullptr) && (gnode->lchild->num_sites == gnode->num_sites))
            continue;
        if ((gnode->lxchild != nullptr) && (gnode->lxchild->num_sites == gnode->num_sites))
            continue;
        if ((gnode->rchild != nullptr) && (gnode->rchild->num_sites == gnode->num_sites))
            continue;
        if ((gnode->rxchild != nullptr) && (gnode->rxchild->num_sites == gnode->num_sites))
            continue;

        usi_t site = gnode->base->site;
        abc_t before_id = site::get_before_id(site);
        for (abc_t after_id : edges[before_id])
        {
            uai_t action_id = action::combine_after_id(site, after_id);
            bool syncope = (after_id == site_space->emp_id);
            float delta = 0.0;
            for (auto order : gnode->linked_words)
            {
                auto word = tnode->words[order];
                if (syncope && (word->id_seq.size() == 3))
                {
                    delta = 9999999999.9;
                    break;
                }
                Word *new_word;
                apply_action(new_word, word, action_id);
                SPDLOG_TRACE("  new word {0} old word {1} order {2}", new_word->str(), word->str(), order);
                // delta += new_word->dists.get_value(order) - word->dists.get_value(order);
                delta += word_space->safe_get_dist(new_word, order) - word_space->safe_get_dist(word, order);
            }
            if (delta < dist_threshold)
                aa.push_back(action_id);
        }
    }
}

inline bool ActionSpace::match(abc_t idx, abc_t target)
{
    if (target == site_space->any_id)
        return ((idx != site_space->sot_id) && (idx != site_space->eot_id));
    return (idx == target);
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
    bool syncope = (after_id == site_space->emp_id);
    int n = id_seq.size();
    new_id_seq.push_back(site_space->sot_id);
    for (int i = 1; i < n - 1; i++)
    {
        bool applied = (id_seq.at(i) == before_id);
        if (applied && (pre_id != NULL_ABC))
        {
            if ((i < 1) || (!match(id_seq.at(i - 1), pre_id)))
                applied = false;
            if (applied && (d_pre_id != NULL_ABC))
                if ((i < 2) || (!match(id_seq.at(i - 2), d_pre_id)))
                    applied = false;
        }
        if (applied && (post_id != NULL_ABC))
        {
            if ((i > n - 2) || (!match(id_seq.at(i + 1), post_id)))
                applied = false;
            if (applied && (d_post_id != NULL_ABC))
                if ((i > n - 3) || (!match(id_seq.at(i + 2), d_post_id)))
                    applied = false;
        }
        if (applied)
            if (syncope)
                continue;
            else
                new_id_seq.push_back(after_id);
        else
            new_id_seq.push_back(id_seq.at(i));
    }
    new_id_seq.push_back(site_space->eot_id);
    return new_id_seq;
}

void ActionSpace::apply_action(Word *&output, Word *word, uai_t action_id)
{
    // Should never deal with stop action here.
    assert(action_id != action::STOP);

    if (word->neighbors.if_contains(action_id, [&output](Word *const &value) { output = value; }))
        return;

    auto new_id_seq = apply_action(word->id_seq, action_id);
    word_space->get_word(output, new_id_seq);

    // NOTE(j_luo) No need to do anything if the key exists -- `get_word` ensures the right Word object is returned.
    word->neighbors.try_emplace_l(
        action_id, [](Word *value) {}, output);
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
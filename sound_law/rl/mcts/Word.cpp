#include "Word.hpp"
#include "Site.hpp"

Word::Word(const IdSeq &id_seq,
           const std::vector<SiteNode *> &site_roots,
           int order,
           float dist,
           bool done) : id_seq(id_seq),
                        site_roots(site_roots),
                        done(done)
{
    // dists[order] = dist;
    orders.push_back(order);
    dists.push_back(dist);
}

size_t Word::size() { return id_seq.size(); }

void Word::debug()
{
    std::cerr << "Debug Word:\n";
    for (abc_t i : id_seq)
        std::cerr << i << ' ';
    std::cerr << '\n';
}

WordSpace::WordSpace(
    SiteSpace *site_space,
    const std::vector<std::vector<float>> &dist_mat,
    float ins_cost,
    const VocabIdSeq &end_ids) : site_space(site_space),
                                 dist_mat(dist_mat),
                                 ins_cost(ins_cost)
{
    for (size_t order = 0; order < end_ids.size(); order++)
        end_words.emplace_back(get_word(end_ids.at(order), order, true));
}

Word *WordSpace::get_word(const IdSeq &id_seq, int order, bool is_end)
{
    // Obtain the read lock for membership test.
    {
        boost::shared_lock_guard<boost::shared_mutex> lock(words_mtx);
        if (words.find(id_seq) != words.end())
        {
            auto word = words.at(id_seq);
            // Compute the edit dist against the right order.
            get_dist(word, order);
            // auto &dists = word->dists;
            // if (dists.find(order) == dists.end())
            // {
            //     dists[order] = is_end ? 0.0 : get_edit_dist(id_seq, end_words.at(order)->id_seq);
            // }
            return words.at(id_seq);
        }
    }

    int n = id_seq.size();
    std::vector<SiteNode *> site_roots = std::vector<SiteNode *>();
    // Start with 1 and end with n - 1 since both ends are padded.
    for (int i = 1; i < n - 1; i++)
    {
        abc_t before = id_seq.at(i);
        abc_t pre_id = (i > 0) ? id_seq.at(i - 1) : NULL_ABC;
        abc_t d_pre_id = (i > 1) ? id_seq.at(i - 2) : NULL_ABC;
        abc_t post_id = (i < n - 1) ? id_seq.at(i + 1) : NULL_ABC;
        abc_t d_post_id = (i < n - 2) ? id_seq.at(i + 2) : NULL_ABC;
        site_roots.push_back(site_space->get_node(before, pre_id, d_pre_id, post_id, d_post_id));
    }
    float dist = is_end ? 0.0 : get_edit_dist(id_seq, end_words.at(order)->id_seq);
    Word *word = new Word(id_seq, site_roots, order, dist, is_end);
    // Obtain the write lock. Release the memeory if it has already been created.
    boost::lock_guard<boost::shared_mutex> lock(words_mtx);
    if (words.find(id_seq) == words.end())
    {
        words[id_seq] = word;
        return word;
    }
    else
    {
        delete word;
        return words.at(id_seq);
    }
}

Word *WordSpace::apply_action(Word *word, uai_t action_id, int order)
{
    ActionMap<Word *> &neighbors = word->neighbors;
    // Return cache if it exists. Obtain the read lock first.
    {
        boost::shared_lock_guard<boost::shared_mutex> lock(word->neighbor_mtx);
        if (neighbors.find(action_id) != neighbors.end())
            return neighbors.at(action_id);
    }

    // Create new word if necessary and cache it as the neighbor.
    auto new_word = apply_action_no_lock(word, action_id, order);
    // Obtain the write lock -- no need to release anything here since it should be taken care of by `get_word`.
    boost::lock_guard<boost::shared_mutex> lock(word->neighbor_mtx);
    neighbors[action_id] = new_word;
    return new_word;
}

inline bool WordSpace::match(abc_t idx, abc_t target)
{
    if (target == site_space->any_id)
        return ((idx != site_space->sot_id) && (idx != site_space->eot_id));
    return (idx == target);
}

Word *WordSpace::apply_action_no_lock(Word *word, uai_t action_id, int order)
{
    // Should never deal with stop action here.
    assert(action_id != action::STOP);

    const IdSeq &id_seq = word->id_seq;
    IdSeq new_id_seq = std::vector<abc_t>();
    abc_t before_id = action::get_before_id(action_id);
    abc_t after_id = action::get_after_id(action_id);
    abc_t pre_id = action::get_pre_id(action_id);
    abc_t d_pre_id = action::get_d_pre_id(action_id);
    abc_t post_id = action::get_post_id(action_id);
    abc_t d_post_id = action::get_d_post_id(action_id);
    bool epenthesis = (after_id == site_space->emp_id);
    int n = word->size();
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
            if (epenthesis)
                continue;
            else
                new_id_seq.push_back(after_id);
        else
            new_id_seq.push_back(id_seq.at(i));
    }
    new_id_seq.push_back(site_space->eot_id);
    return get_word(new_id_seq, order, false);
}

size_t WordSpace::size() { return words.size(); }

float WordSpace::get_edit_dist(const IdSeq &seq1, const IdSeq &seq2)
{
    size_t l1 = seq1.size();
    size_t l2 = seq2.size();
    float **dist = (float **)malloc((l1 + 1) * sizeof(float **));
    for (size_t i = 0; i < l1 + 1; ++i)
        dist[i] = (float *)malloc((l2 + 1) * sizeof(float *));

    for (size_t i = 0; i < l1 + 1; ++i)
        dist[i][0] = i * ins_cost;
    for (size_t i = 0; i < l2 + 1; ++i)
        dist[0][i] = i * ins_cost;

    float sub_cost;
    for (int i = 1; i < l1 + 1; ++i)
        for (int j = 1; j < l2 + 1; ++j)
        {
            sub_cost = dist_mat[seq1.at(i - 1)][seq2.at(j - 1)];
            dist[i][j] = std::min(dist[i - 1][j - 1] + sub_cost, std::min(dist[i - 1][j], dist[i][j - 1]) + ins_cost);
        }
    float ret = dist[l1][l2];
    for (size_t i = 0; i < l1 + 1; ++i)
        free(dist[i]);
    free(dist);
    return ret;
};

float Word::get_dist(int order)
{
    auto o_it = orders.begin();
    auto d_it = dists.begin();
    while (o_it != orders.end())
    {
        if (*o_it == order)
            return *d_it;
        o_it++;
        d_it++;
    }
    throw std::out_of_range("Cannot find this order for this word.");
}

float WordSpace::get_dist(Word *word, int order)
{
    try
    {
        return word->get_dist(order);
    }
    catch (std::out_of_range e)
    {
        word->orders.push_back(order);
        auto dist = get_edit_dist(word->id_seq, end_words.at(order)->id_seq);
        word->dists.push_back(dist);
        return dist;
    }
    // if (dists.find(order) == dists.end())
    // {
    //     dists[order] = get_edit_dist(word->id_seq, end_words.at(order)->id_seq);
    // }
    // return dists[order];
}

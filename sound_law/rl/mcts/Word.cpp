#include "Word.hpp"
#include "Site.hpp"

Word::Word(const IdSeq &id_seq,
           const std::vector<SiteNode *> &site_roots,
           float dist,
           bool done) : id_seq(id_seq),
                        site_roots(site_roots),
                        dist(dist),
                        done(done) {}

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
        end_words.push_back(get_word(end_ids.at(order), order, true));
}

Word *WordSpace::get_word(const IdSeq &id_seq, int order, bool is_end)
{
    if (words.find(id_seq) == words.end())
    {
        int n = id_seq.size();
        std::vector<SiteNode *> site_roots = std::vector<SiteNode *>();
        for (int i = 0; i < n; i++)
        {
            abc_t before = id_seq.at(i);
            abc_t pre_id = (i > 0) ? id_seq.at(i - 1) : NULL_abc;
            abc_t d_pre_id = (i > 1) ? id_seq.at(i - 2) : NULL_abc;
            abc_t post_id = (i < n - 1) ? id_seq.at(i + 1) : NULL_abc;
            abc_t d_post_id = (i < n - 2) ? id_seq.at(i + 2) : NULL_abc;
            site_roots.push_back(site_space->get_node(before, pre_id, d_pre_id, post_id, d_post_id));
        }
        float dist = is_end ? 0.0 : get_edit_dist(id_seq, end_words.at(order)->id_seq);
        Word *word = new Word(id_seq, site_roots, dist, is_end);
        words[id_seq] = word;
        return word;
    }
    else
    {
        std::cerr << "reusing word\n";
        return words.at(id_seq);
    }
}

Word *WordSpace::apply_action(Word *word, const Action &action, int order)
{
    boost::unordered_map<Action, Word *> &neighbors = word->neighbors;
    // Return cache if it exists.
    if (neighbors.find(action) != neighbors.end())
        return neighbors.at(action);

    // Compute the new id seq.
    const IdSeq &id_seq = word->id_seq;
    IdSeq new_id_seq = std::vector<abc_t>();
    abc_t before_id = action.at(0);
    abc_t after_id = action.at(1);
    abc_t pre_id = action.at(2);
    abc_t d_pre_id = action.at(3);
    abc_t post_id = action.at(4);
    abc_t d_post_id = action.at(5);
    size_t n = word->size();
    for (size_t i = 0; i < n; i++)
    {
        new_id_seq.push_back(id_seq.at(i));
        if (id_seq.at(i) == before_id)
        {
            if (pre_id != NULL_abc)
            {
                if ((i == 0) || (id_seq.at(i - 1) != pre_id))
                    continue;
                if (d_pre_id != NULL_abc)
                    if ((i <= 1) || (id_seq.at(i - 2) != d_pre_id))
                        continue;
            }
            if (post_id != NULL_abc)
            {
                if ((i == n - 1) || (id_seq.at(i + 1) != post_id))
                    continue;
                if (d_post_id != NULL_abc)
                    if ((i >= n - 2) || (id_seq.at(i + 2) != d_post_id))
                        continue;
            }
            new_id_seq[i] = after_id;
        }
    }

    // Create new word if necessary and cache it as the neighbor.
    Word *new_word = get_word(new_id_seq, order, false);
    neighbors[action] = new_word;
    return new_word;
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
    bool use_phono_edit_dist = (dist_mat.size() > 0);
    for (size_t i = 1; i < l1 + 1; ++i)
        for (size_t j = 1; j < l2 + 1; ++j)
        {
            if (use_phono_edit_dist)
            {
                sub_cost = dist_mat[seq1[i - 1]][seq2[j - 1]];
            }
            else
            {
                sub_cost = seq1[i - 1] == seq2[j - 1] ? 0 : 1;
            }
            dist[i][j] = std::min(dist[i - 1][j - 1] + sub_cost, std::min(dist[i - 1][j], dist[i][j - 1]) + ins_cost);
        }
    float ret = dist[l1][l2];
    for (size_t i = 0; i < l1 + 1; ++i)
        free(dist[i]);
    free(dist);
    return ret;
};
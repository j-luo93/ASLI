#include "word.hpp"

Word::Word(const IdSeq &id_seq,
           const vec<SiteNode *> &site_roots,
           size_t dt_size) : id_seq(id_seq),
                             site_roots(site_roots),
                             dists(DistTable(dt_size)){};

std::string Word::str()
{
    std::string out = "";
    for (size_t i = 1; i < id_seq.size() - 2; i++)
        out += std::to_string(id_seq.at(i)) + ",";
    out += std::to_string(id_seq.at(id_seq.size() - 2));
    return out;
}

WordSpace::WordSpace(SiteSpace *site_space,
                     const vec<vec<float>> &dist_mat,
                     float ins_cost) : site_space(site_space),
                                       dist_mat(dist_mat),
                                       ins_cost(ins_cost){};

size_t WordSpace::size() const { return words.size(); }

void WordSpace::get_word(Word *&output, const IdSeq &id_seq, size_t dt_size)
{
    // timer.start("get_word");
    dt_size = end_words.empty() ? dt_size : end_words.size();
    assert(dt_size > 0);

    if (words.if_contains(id_seq, [&output](Word *const &value) { output = value; }))
    {
        // timer.end("get_word");
        return;
    }

    size_t n = id_seq.size();
    auto site_roots = vec<SiteNode *>(n - 2);
    for (int i = 1; i < n - 1; i++)
    {
        abc_t before_id = id_seq[i];
        abc_t pre_id = (i > 0) ? id_seq[i - 1] : NULL_ABC;
        abc_t d_pre_id = (i > 1) ? id_seq[i - 2] : NULL_ABC;
        abc_t post_id = (i < n - 1) ? id_seq[i + 1] : NULL_ABC;
        abc_t d_post_id = (i < n - 2) ? id_seq[i + 2] : NULL_ABC;
        site_space->get_node(site_roots[i - 1], before_id, pre_id, d_pre_id, post_id, d_post_id);
    }
    auto word = new Word(id_seq, site_roots, dt_size);
    output = word;
    words.try_emplace_l(
        id_seq, [&output, word](Word *&value) { output = value; delete word; }, word);
}

void WordSpace::get_words(Pool *tp, vec<Word *> &outputs, const vec<IdSeq> &inputs, bool unique, size_t dt_size)
{
    parallel_apply<true>(
        tp,
        [this, dt_size](Word *&output, const IdSeq &input) { get_word(output, input, dt_size); },
        outputs,
        inputs);
}

inline float WordSpace::get_edit_dist(Word *word1, Word *word2)
{
    return get_edit_dist(word1->id_seq, word2->id_seq);
}

inline float WordSpace::get_edit_dist(const IdSeq &seq1, const IdSeq &seq2)
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
    for (size_t i = 1; i < l1 + 1; ++i)
        for (size_t j = 1; j < l2 + 1; ++j)
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

void WordSpace::set_end_words(const vec<Word *> &words)
{
    this->end_words = words;
}

float WordSpace::safe_get_dist(Word *word, int order)
{
    std::atomic<float> *ptr = word->dists.locate(order);
    if (*ptr < 0)
        *ptr = get_edit_dist(word, end_words[order]);
    return *ptr;
}
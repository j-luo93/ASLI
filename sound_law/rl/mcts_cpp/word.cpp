#include "word.hpp"

Word::Word(const IdSeq &id_seq) : id_seq(id_seq) {}

float Word::get_edit_dist_at(int order) { return dists[order]; }

WordSpace::WordSpace(const VocabIdSeq &end_ids, const WordSpaceOpt &ws_opt) : opt(ws_opt)
{
    end_words.reserve(end_ids.size());
    for (const auto &id_seq : end_ids)
        end_words.push_back(get_word(id_seq));
}

Word *WordSpace::get_word(const IdSeq &id_seq)
{
    Word *output;
    if (words.if_contains(id_seq, [&output](Word *const &value) { output = value; }))
        return output;

    auto word = new Word(id_seq);
    output = word;
    // Delete the newly-constructed Word instance if it has been constructed by another thread.
    words.try_emplace_l(
        id_seq, [&output, word](Word *&value) { output = value; delete word; }, word);
    return output;
}

void WordSpace::set_edit_dist_at(Word *word, int order)
{
    if (word->dists.if_contains(order, [](const float dist) {}))
        return;

    auto dist = get_edit_dist(word->id_seq, end_words[order]->id_seq);
    word->dists.try_emplace_l(
        order, [](float dist) {}, dist);
};

float WordSpace::get_edit_dist(const IdSeq &seq1, const IdSeq &seq2)
{
    size_t l1 = seq1.size();
    size_t l2 = seq2.size();
    float **dist = (float **)malloc((l1 + 1) * sizeof(float **));
    for (size_t i = 0; i < l1 + 1; ++i)
        dist[i] = (float *)malloc((l2 + 1) * sizeof(float *));

    for (size_t i = 0; i < l1 + 1; ++i)
        dist[i][0] = i * opt.ins_cost;
    for (size_t i = 0; i < l2 + 1; ++i)
        dist[0][i] = i * opt.ins_cost;

    float sub_cost;
    for (size_t i = 1; i < l1 + 1; ++i)
        for (size_t j = 1; j < l2 + 1; ++j)
        {
            sub_cost = opt.dist_mat[seq1.at(i - 1)][seq2.at(j - 1)];
            dist[i][j] = std::min(dist[i - 1][j - 1] + sub_cost, std::min(dist[i - 1][j], dist[i][j - 1]) + opt.ins_cost);
        }
    float ret = dist[l1][l2];
    for (size_t i = 0; i < l1 + 1; ++i)
        free(dist[i]);
    free(dist);
    return ret;
}
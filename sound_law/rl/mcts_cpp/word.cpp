#include "word.hpp"

Word::Word(const IdSeq &id_seq,
           const IdSeq &vowel_seq,
           const vec<size_t> &id2vowel) : id_seq(id_seq),
                                          vowel_seq(vowel_seq),
                                          id2vowel(id2vowel) {}

float Word::get_edit_dist_at(int order) const { return dists.at(order); }

WordSpace::WordSpace(const WordSpaceOpt &ws_opt, const VocabIdSeq &end_ids) : opt(ws_opt), end_words(get_words(end_ids)) {}

Word *WordSpace::get_word(const IdSeq &id_seq)
{
    Word *output;
    if (words.if_contains(id_seq, [&output](Word *const &value) { output = value; }))
        return output;

    size_t n = id_seq.size();
    auto vowel_seq = vec<abc_t>();
    auto id2vowel = vec<size_t>();
    vowel_seq.reserve(n);
    id2vowel.reserve(n);

    vowel_seq.push_back(id_seq[0]);
    id2vowel.push_back(0);
    for (size_t i = 1; i < id_seq.size() - 1; ++i)
    {
        abc_t unit = id_seq[i];
        if (opt.is_vowel[unit])
        {
            id2vowel.push_back(vowel_seq.size());
            vowel_seq.push_back(unit);
        }
        else
            id2vowel.push_back(0);
    }

    auto word = new Word(id_seq, vowel_seq, id2vowel);

    output = word;
    // Delete the newly-constructed Word instance if it has been constructed by another thread.
    words.try_emplace_l(
        id_seq, [&output, word](Word *&value) { output = value; delete word; }, word);
    return output;
}

vec<Word *> WordSpace::get_words(const VocabIdSeq &vocab)
{
    auto words = vec<Word *>();
    words.reserve(vocab.size());
    for (const auto &id_seq : vocab)
        words.push_back(get_word(id_seq));
    return words;
}

void WordSpace::set_edit_dist_at(Word *word, int order) const
{
    if (word->dists.if_contains(order, [](const float dist) {}))
        return;

    auto dist = get_edit_dist(word->id_seq, end_words[order]->id_seq);
    word->dists.try_emplace_l(
        order, [](float &dist) {}, dist);
};

float WordSpace::get_edit_dist(const IdSeq &seq1, const IdSeq &seq2) const
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

size_t WordSpace::size() const { return words.size(); }
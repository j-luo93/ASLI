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
    if (word->dists.if_contains(order, [](const float &dist) {}))
        return;

    float dist;
    Alignment almt;
    if (opt.use_alignment)
        dist = get_edit_dist(word->id_seq, end_words[order]->id_seq, almt);
    else
        dist = get_edit_dist(word->id_seq, end_words[order]->id_seq);
    word->dists.try_emplace_l(
        order, [](float &dist) {}, dist);
    if (opt.use_alignment)
        word->almts.try_emplace_l(
            order, [](Alignment &almt) {}, almt);
};

float WordSpace::get_edit_dist(const IdSeq &seq1, const IdSeq &seq2) const
{
    auto almt = Alignment();
    return get_edit_dist(seq1, seq2, almt);
}

enum class EditOp : int
{
    INSERTION,
    DELETION,
    SUBSTITUTION,
};

float WordSpace::get_edit_dist(const IdSeq &seq1, const IdSeq &seq2, Alignment &almt) const
{
    size_t l1 = seq1.size();
    size_t l2 = seq2.size();
    float **dist = (float **)malloc((l1 + 1) * sizeof(float **));
    for (size_t i = 0; i < l1 + 1; ++i)
        dist[i] = (float *)malloc((l2 + 1) * sizeof(float *));
    // This records what is the best op.
    EditOp **best = (EditOp **)malloc((l1 + 1) * sizeof(EditOp **));
    for (size_t i = 0; i < l1 + 1; ++i)
        best[i] = (EditOp *)malloc((l2 + 1) * sizeof(EditOp *));

    for (size_t i = 0; i < l1 + 1; ++i)
    {
        dist[i][0] = i * opt.ins_cost;
        best[i][0] = EditOp::INSERTION;
    }
    for (size_t i = 0; i < l2 + 1; ++i)
    {
        dist[0][i] = i * opt.ins_cost;
        best[0][i] = EditOp::DELETION;
    }

    float cost, icost, dcost;
    for (size_t i = 1; i < l1 + 1; ++i)
        for (size_t j = 1; j < l2 + 1; ++j)
        {
            cost = opt.dist_mat[seq1[i - 1]][seq2[j - 1]] + dist[i - 1][j - 1];
            dist[i][j] = cost;
            best[i][j] = EditOp::SUBSTITUTION;
            icost = dist[i - 1][j] + opt.ins_cost;
            if (icost < cost)
            {
                dist[i][j] = icost;
                best[i][j] = EditOp::INSERTION;
                cost = icost;
            }
            dcost = dist[i][j - 1] + opt.ins_cost;
            if (dcost < cost)
            {
                dist[i][j] = dcost;
                best[i][j] = EditOp::DELETION;
            }
            // dist[i][j] = std::min(dist[i - 1][j - 1] + scost, std::min(dist[i - 1][j], dist[i][j - 1]) + opt.ins_cost);
        }
    float ret = dist[l1][l2];
    // Backtrack to get the best alignment.
    size_t best_i = l1;
    size_t best_j = l2;
    EditOp op;
    // Get the (reversed) list of edit ops first.
    auto ops = vec<EditOp>();
    ops.reserve(l1 + l2);
    while (true)
    {
        op = best[best_i][best_j];
        ops.push_back(op);
        switch (op)
        {
        case EditOp::INSERTION:
            --best_i;
            break;
        case EditOp::DELETION:
            --best_j;
            break;
        case EditOp::SUBSTITUTION:
            --best_i;
            --best_j;
            break;
        }
        assert(best_i >= 0);
        assert(best_j >= 0);
        if ((best_i == 0) && (best_j == 0))
            break;
    }
    // Go backwards and find the aligned indices.
    size_t pos1 = 0;
    size_t pos2 = 0;
    size_t almt_pos = 0;
    auto &pos_seq1 = almt.pos_seq1;
    pos_seq1.reserve(l1);
    auto &pos_seq2 = almt.pos_seq2;
    pos_seq2.reserve(l2);
    auto &aligned_pos = almt.aligned_pos;
    aligned_pos.reserve(l1);
    for (auto it = ops.rbegin(); it != ops.rend(); ++it)
    {
        switch (*it)
        {
        case EditOp::INSERTION:
            ++pos1;
            pos_seq1.push_back(almt_pos++);
            aligned_pos.push_back(static_cast<int>(alignment::INSERTED));
            break;
        case EditOp::DELETION:
            ++pos2;
            pos_seq2.push_back(almt_pos++);
            break;
        case EditOp::SUBSTITUTION:
            ++pos1;
            aligned_pos.push_back(pos2++);
            pos_seq1.push_back(almt_pos);
            pos_seq2.push_back(almt_pos++);
            break;
        }
    }
    assert(pos_seq1.size() == l1);
    assert(pos_seq2.size() == l2);
    assert(aligned_pos.size() == l1);

    for (size_t i = 0; i < l1 + 1; ++i)
        free(dist[i]);
    free(dist);
    for (size_t i = 0; i < l1 + 1; ++i)
        free(best[i]);
    free(best);
    return ret;
}

size_t WordSpace::size() const { return words.size(); }

const Alignment &Word::get_almt_at(int order) const { return almts.at(order); }

float WordSpace::get_misalignment_score(const Word *word, int order, size_t position, abc_t after_id) const
{
    if (!opt.use_alignment)
        return 0.0;

    assert(opt.use_alignment);
    const auto &almt = word->get_almt_at(order);
    const auto c1 = word->id_seq[position];
    const auto aligned_pos = almt.aligned_pos[position];
    if (aligned_pos == alignment::INSERTED)
        return ((after_id == 4) || (after_id == abc::NONE)) ? opt.ins_cost : 0.0;
    assert(aligned_pos < end_words[order]->id_seq.size());
    const auto c2 = end_words[order]->id_seq[aligned_pos];
    if (after_id == 4)
        return opt.dist_mat[c1][c2] - opt.ins_cost;
    if (after_id == abc::NONE)
        return opt.dist_mat[c1][c2];
    else
        return opt.dist_mat[c1][c2] - opt.dist_mat[after_id][c2];
}
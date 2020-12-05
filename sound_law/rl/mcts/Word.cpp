#include "Word.hpp"

Word::Word(const IdSeq &id_seq) : id_seq(id_seq) {}
size_t Word::size() { return id_seq.size(); }
void Word::debug()
{
    std::cerr << "Debug Word:\n";
    for (abc_t i : id_seq)
        std::cerr << i << ' ';
    std::cerr << '\n';
}

Word *WordSpace::get_word(const IdSeq &id_seq)
{
    if (words.find(id_seq) == words.end())
    {
        Word *word = new Word(id_seq);
        words[id_seq] = word;
        return word;
    }
    else
    {
        std::cerr << "reusing word\n";
        return words.at(id_seq);
    }
}

Word *WordSpace::apply_action(Word *word, const Action &action)
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
    Word *new_word = get_word(new_id_seq);
    neighbors[action] = new_word;
    return new_word;
}

size_t WordSpace::size() { return words.size(); }

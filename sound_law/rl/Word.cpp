#include <Word.h>

Word::Word(const IdSeq &id_seq)
{
    this->key = get_key(id_seq);
    this->uni_keys = unordered_set<abc_t>();
    unordered_set<abc_t> pre = unordered_set<abc_t>();
    this->pre_keys = unordered_map<abc_t, unordered_set<abc_t>>();
    for (size_t i = 0; i < id_seq.size(); ++i)
    {
        abc_t id = id_seq[i];
        this->uni_keys.insert(id);
        if (i > 0)
            pre.insert(id);
    }
    for (abc_t const id : pre)
        this->pre_keys[id] = unordered_set<abc_t>();
    for (size_t i = 1; i < id_seq.size(); ++i)
    {
        this->pre_keys[id_seq[i]].insert(id_seq[i - 1]);
    }
}
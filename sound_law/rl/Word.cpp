#include <Word.h>

Word::Word(const IdSeq &id_seq)
{
    this->key = get_key(id_seq);
    this->uni_keys = unordered_set<long>();
    unordered_set<long> pre = unordered_set<long>();
    this->pre_keys = unordered_map<long, unordered_set<long>>();
    for (long i = 0; i < id_seq.size(); ++i)
    {
        long id = id_seq[i];
        this->uni_keys.insert(id);
        if (i > 0)
            pre.insert(id);
    }
    for (long const id : pre)
        this->pre_keys[id] = unordered_set<long>();
    for (long i = 1; i < id_seq.size(); ++i)
    {
        this->pre_keys[id_seq[i]].insert(id_seq[i - 1]);
    }
}
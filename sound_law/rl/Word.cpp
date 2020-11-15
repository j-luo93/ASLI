#include <Word.h>

Word::Word(const IdSeq &id_seq)
{
    this->key = get_key(id_seq);
    this->uni_keys = unordered_set<long>(id_seq.begin(), id_seq.end());
}
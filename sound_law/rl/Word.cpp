#include <Word.h>

Word::Word(const IdSeq &id_seq, const WordKey &key)
{
    this->key = key;
    this->sites = vector<Site>();
    const IdSeq::const_iterator it = id_seq.begin();
    // Start with 1 and end with length - 1 since both ends are padded.
    for (int i = 1; i < id_seq.size() - 1; ++i)
    {
        abc_t before_id = id_seq.at(i);
        int start = max(1, i - 2);
        int end = min(i + 3, (int)id_seq.size() - 1);
        vector<abc_t> pre_cond = vector<abc_t>(it + start, it + i);
        vector<abc_t> post_cond = vector<abc_t>(it + i + 1, it + end);
        SiteKey key = get_site_key(before_id, pre_cond, post_cond);
        this->sites.push_back(Site(before_id, pre_cond, post_cond, key));
    }
}
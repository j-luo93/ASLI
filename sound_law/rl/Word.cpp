#include <Word.h>
#include <Site.h>

VocabIdSeq Word::end_words = VocabIdSeq();

void Word::set_end_words(const VocabIdSeq &vocab_i)
{
    Word::end_words = vocab_i;
}

Word::Word(const IdSeq &id_seq, const WordKey &key, size_t order) : id_seq(id_seq), order(order)
{
    this->key = key;
    this->site_roots = vector<SiteNode *>();
    const IdSeq::const_iterator it = id_seq.begin();
    // Start with 1 and end with length - 1 since both ends are padded.
    for (int i = 1; i < id_seq.size() - 1; ++i)
    {
        abc_t before_id = id_seq.at(i);
        abc_t pre_id = (i > 1) ? id_seq.at(i - 1) : NULL_abc;
        abc_t d_pre_id = (i > 2) ? id_seq.at(i - 2) : NULL_abc;
        abc_t post_id = (i < id_seq.size() - 2) ? id_seq.at(i + 1) : NULL_abc;
        abc_t d_post_id = (i < id_seq.size() - 3) ? id_seq.at(i + 2) : NULL_abc;
        // This will get the unique site node associated with this site, and generate a site graph.
        SiteNode *root = SiteNode::get_site_node(before_id, pre_id, d_pre_id, post_id, d_post_id, this);
        this->site_roots.push_back(root);
    }
}

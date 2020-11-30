#include <Site.h>

Site::Site(abc_t before_id,
           const vector<abc_t> &pre_cond,
           const vector<abc_t> &post_cond,
           const SiteKey &key) : before_id(before_id),
                                 pre_cond(pre_cond),
                                 post_cond(post_cond),
                                 key(key) {}

SiteGraph::SiteGraph()
{
    this->nodes = unordered_map<SiteKey, SiteNode *>();
}

SiteNode::SiteNode(const Site &site) : site(site)
{
    this->children = vector<SiteNode *>();
}

SiteNode::SiteNode(abc_t before_id,
                   const vector<abc_t> &pre_cond,
                   const vector<abc_t> &post_cond,
                   const SiteKey &key) : site(Site(before_id, pre_cond, post_cond, key))
{
    this->children = vector<SiteNode *>();
}

void SiteGraph::add_site(const Site &new_site, SiteNode *parent)
{
    const SiteKey &key = new_site.key;
    if (this->nodes.find(key) == this->nodes.end())
    {
        SiteNode *new_node = new SiteNode(new_site);
        this->nodes[key] = new_node;
    }
    this->add_site(key, parent);
}

void SiteGraph::add_site(const SiteKey &key, SiteNode *parent)
{
    SiteNode *node = this->nodes[key];
    ++node->num_sites;
    if (parent != nullptr)
    {
        parent->children.push_back(node);
        ++node->in_degree;
    }
    const Site &site = node->site;
    if (site.pre_cond.size() > 0)
    {
        vector<abc_t> new_pre_cond = vector<abc_t>(site.pre_cond.begin() + 1, site.pre_cond.end());
        const SiteKey &key = get_site_key(site.before_id, new_pre_cond, site.post_cond);
        this->add_site(site.before_id, new_pre_cond, site.post_cond, key, node);
    }
    if (site.post_cond.size() > 0)
    {
        vector<abc_t> new_post_cond = vector<abc_t>(site.post_cond.begin(), site.post_cond.end() - 1);
        const SiteKey &key = get_site_key(site.before_id, site.pre_cond, new_post_cond);
        this->add_site(site.before_id, site.pre_cond, new_post_cond, key, node);
    }
}

void SiteGraph::add_site(abc_t before_id, const vector<abc_t> &pre_cond, const vector<abc_t> &post_cond, const SiteKey &key, SiteNode *parent)
{
    if (this->nodes.find(key) == this->nodes.end())
    {
        SiteNode *new_node = new SiteNode(before_id, pre_cond, post_cond, key);
        this->nodes[key] = new_node;
    }
    this->add_site(key, parent);
}

vector<SiteNode *> SiteGraph::get_sources()
{
    vector<SiteNode *> sources = vector<SiteNode *>();
    for (auto const item : this->nodes)
    {
        SiteNode *node = item.second;
        if (node->in_degree == 0)
            sources.push_back(node);
    }
    assert(!sources.empty());
    return sources;
}

SiteGraph::~SiteGraph()
{
    for (auto item : this->nodes)
    {
        delete item.second;
    }
}
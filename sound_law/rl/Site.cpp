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

void SiteGraph::add_site(const Site &new_site, SiteNode *parent)
{
    SiteKey key = new_site.key;
    if (this->nodes.find(key) == this->nodes.end())
    {
        SiteNode *new_node = new SiteNode(new_site);
        this->nodes[key] = new_node;
    }
    // Use this merged node in the graph instead of a new node.
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
        const Site &child_site = Site(site.before_id, new_pre_cond, site.post_cond, key);
        this->add_site(child_site, node);
    }
    if (site.post_cond.size() > 0)
    {
        vector<abc_t> new_post_cond = vector<abc_t>(site.post_cond.begin(), site.post_cond.end() - 1);
        const SiteKey &key = get_site_key(site.before_id, site.pre_cond, new_post_cond);
        const Site &child_site = Site(site.before_id, site.pre_cond, new_post_cond, key);
        this->add_site(child_site, node);
    }
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
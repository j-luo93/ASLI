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
    this->nodes = unordered_map<SiteKey, shared_ptr<SiteNode>>();
}

SiteNode::SiteNode(const Site &site) : site(site)
{
    this->children = vector<shared_ptr<SiteNode>>();
}

void SiteGraph::add_site(const Site &new_site, shared_ptr<SiteNode> parent)
{
    shared_ptr<SiteNode> new_node = make_shared<SiteNode>(SiteNode(new_site));
    SiteKey key = new_node->site.key;
    if (this->nodes.find(key) == this->nodes.end())
        this->nodes[key] = new_node;
    // Use this merged node in the graph instead of a new node.
    shared_ptr<SiteNode> node = this->nodes[key];
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
        SiteKey key = get_site_key(site.before_id, new_pre_cond, site.post_cond);
        Site child_site = Site(site.before_id, new_pre_cond, site.post_cond, key);
        this->add_site(child_site, node);
    }
    if (site.post_cond.size() > 0)
    {
        vector<abc_t> new_post_cond = vector<abc_t>(site.post_cond.begin(), site.post_cond.end() - 1);
        SiteKey key = get_site_key(site.before_id, site.pre_cond, new_post_cond);
        Site child_site = Site(site.before_id, site.pre_cond, new_post_cond, key);
        this->add_site(child_site, node);
    }
}

vector<shared_ptr<SiteNode>> SiteGraph::get_sources()
{
    vector<shared_ptr<SiteNode>> sources = vector<shared_ptr<SiteNode>>();
    for (auto const item : this->nodes)
    {
        shared_ptr<SiteNode> node = item.second;
        if (node->in_degree == 0)
            sources.push_back(node);
    }
    assert(!sources.empty());
    return sources;
}
#include <Site.h>

// Site::Site(abc_t before_id,
//            const vector<abc_t> &pre_cond,
//            const vector<abc_t> &post_cond,
//            const SiteKey &key) : before_id(before_id),
//                                  pre_cond(pre_cond),
//                                  post_cond(post_cond),
//                                  key(key) {}

SiteGraph::SiteGraph()
{
    this->nodes = unordered_map<Site, SiteNode *>();
}

SiteNode::SiteNode(const Site &site) : site(site)
{
    this->children = vector<SiteNode *>();
}

void SiteGraph::add_site(const Site &new_site, SiteNode *parent)
{
    // const SiteKey &key = new_site.key;
    if (this->nodes.find(new_site) == this->nodes.end())
    {
        SiteNode *new_node = new SiteNode(new_site);
        this->nodes[new_site] = new_node;
    }

    SiteNode *node = this->nodes[new_site];
    ++node->num_sites;
    if (parent != nullptr)
    {
        parent->children.push_back(node);
        ++node->in_degree;
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
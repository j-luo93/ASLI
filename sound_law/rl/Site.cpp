#include <Site.h>

unordered_map<Site, SiteNode *> SiteNode::all_nodes = unordered_map<Site, SiteNode *>();

SiteGraph::SiteGraph()
{
    this->nodes = unordered_map<Site, SiteNode *>();
}

inline SiteNode *generate_subgraph(abc_t before_id,
                                   abc_t pre_id,
                                   abc_t d_pre_id,
                                   abc_t post_id,
                                   abc_t d_post_id,
                                   SiteNode *parent = nullptr)
{
    Site site = Site(before_id, pre_id, d_pre_id, post_id, d_post_id);
    if (SiteNode::all_nodes.find(site) == SiteNode::all_nodes.end())
    {
        SiteNode *new_node = new SiteNode();
        new_node->site = site;
        SiteNode::all_nodes[site] = new_node;
        // Generate all its children.
        if (pre_id != NULL_abc)
            if (d_pre_id != NULL_abc)
                new_node->lchild = generate_subgraph(before_id, pre_id, NULL_abc, post_id, d_post_id, new_node);
            else
                new_node->lchild = generate_subgraph(before_id, NULL_abc, NULL_abc, post_id, d_post_id, new_node);
        if (post_id != NULL_abc)
            if (d_post_id != NULL_abc)
                new_node->rchild = generate_subgraph(before_id, pre_id, d_pre_id, post_id, NULL_abc, new_node);
            else
                new_node->rchild = generate_subgraph(before_id, pre_id, d_pre_id, NULL_abc, NULL_abc, new_node);
    }
    return SiteNode::all_nodes.at(site);
}

SiteNode *SiteNode::get_site_node(abc_t before_id, abc_t pre_id, abc_t d_pre_id, abc_t post_id, abc_t d_post_id)
{
    Site site = Site(before_id, pre_id, d_pre_id, post_id, d_post_id);
    unique_lock<mutex> lock(SiteNode::cls_mtx);
    SiteNode *node = generate_subgraph(before_id, pre_id, d_pre_id, post_id, d_post_id);
    lock.unlock();
    return node;
}

void SiteNode::reset()
{
    this->num_sites = 0;
    this->in_degree = 0;
    this->visited = false;
}

void SiteGraph::add_node(SiteNode *node)
{
    if (this->nodes.find(node->site) == this->nodes.end())
        this->nodes[node->site] = node;
    ++node->num_sites;
    if (node->lchild != nullptr)
    {
        ++node->lchild->in_degree;
        this->add_node(node->lchild);
    }
    if (node->rchild != nullptr)
    {
        ++node->rchild->in_degree;
        this->add_node(node->rchild);
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
        item.second->reset();
    }
}
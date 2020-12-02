#include <Site.h>

unordered_map<Site, SiteNode *> SiteNode::all_nodes = unordered_map<Site, SiteNode *>();
mutex SiteNode::cls_mtx;

SiteGraph::SiteGraph()
{
    this->nodes = unordered_map<Site, SiteNodeWithStats *>();
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

SiteNodeWithStats::SiteNodeWithStats(SiteNode *node) : base(node)
{
    this->num_sites = 0;
    this->in_degree = 0;
    this->visited = false;
}

inline SiteNodeWithStats *SiteGraph::get_wrapped_node(SiteNode *base)
{
    SiteNodeWithStats *node;
    if (this->nodes.find(base->site) != this->nodes.end())
        node = this->nodes.at(base->site);
    else
    {
        node = new SiteNodeWithStats(base);
        this->nodes[base->site] = node;
    }
    return node;
}

void *SiteGraph::add_node(SiteNode *base, SiteNode *parent)
{
    // Generate all wrapped nodes first.
    vector<SiteNodeWithStats *> queue = vector<SiteNodeWithStats *>();
    SiteNodeWithStats *base_node = this->get_wrapped_node(base);
    base_node->visited = true;
    queue.push_back(base_node);
    size_t i = 0;
    while (i < queue.size())
    {
        SiteNodeWithStats *snode = queue.at(i);
        const SiteNode *node = snode->base;
        if (node->lchild != nullptr)
        {
            if (snode->lchild == nullptr)
                snode->lchild = this->get_wrapped_node(node->lchild);
            snode->lchild->in_degree++;
            if (!snode->lchild->visited)
            {
                queue.push_back(snode->lchild);
                snode->lchild->visited = true;
            }
        }
        if (node->rchild != nullptr)
        {
            if (snode->rchild == nullptr)
                snode->rchild = this->get_wrapped_node(node->rchild);
            snode->rchild->in_degree++;
            if (!snode->rchild->visited)
            {
                queue.push_back(snode->rchild);
                snode->rchild->visited = true;
            }
        }
        ++i;
    }
    // Increment every wrapped node and reset visited.
    for (SiteNodeWithStats *snode : queue)
    {
        ++snode->num_sites;
        snode->visited = false;
    }
}

vector<SiteNodeWithStats *> SiteGraph::get_sources()
{
    vector<SiteNodeWithStats *> sources = vector<SiteNodeWithStats *>();
    for (auto const item : this->nodes)
    {
        SiteNodeWithStats *node = item.second;
        if (node->in_degree == 0)
            sources.push_back(node);
    }
    assert(!sources.empty());
    return sources;
}

SiteGraph::~SiteGraph()
{
    for (auto const &item : this->nodes)
        delete item.second;
}
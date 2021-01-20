#include "site.hpp"

SiteNode::SiteNode(usi_t site) : site(site) {}

SiteSpace::SiteSpace(abc_t sot_id, abc_t eot_id, abc_t any_id, abc_t emp_id, abc_t syl_eot_id) : sot_id(sot_id),
                                                                                                 eot_id(eot_id),
                                                                                                 any_id(any_id),
                                                                                                 emp_id(emp_id),
                                                                                                 syl_eot_id(syl_eot_id) {}

size_t SiteSpace::size() const { return nodes.size(); }

void SiteSpace::get_node(SiteNode *&output, usi_t site)
{
    auto before_id = site::get_before_id(site);
    auto pre_id = site::get_pre_id(site);
    auto d_pre_id = site::get_d_pre_id(site);
    auto post_id = site::get_post_id(site);
    auto d_post_id = site::get_d_post_id(site);
    get_node(output, before_id, pre_id, d_pre_id, post_id, d_post_id);
}

inline void SiteSpace::get_node(SiteNode *&output, abc_t before_id, abc_t pre_id, abc_t d_pre_id, abc_t post_id, abc_t d_post_id)
{
    auto site = site::combine(pre_id, d_pre_id, post_id, d_post_id, before_id);
    // NOTE(j_luo) const T* & is different from T *const &. See https://stackoverflow.com/questions/3316562/const-pointer-assign-to-a-pointer.
    if (nodes.if_contains(site, [&output](SiteNode *const &value) { output = value; }))
        return;

    auto new_node = new SiteNode(site);
    output = new_node;
    if (!nodes.try_emplace_l(
            site, [&output, &new_node](SiteNode *&value) { output = value; delete new_node; }, new_node))
        return;

    SPDLOG_TRACE("adding site to nodes {}", site::str(site));
    if (pre_id != NULL_ABC)
    {
        if (d_pre_id != NULL_ABC)
        {
            if ((d_pre_id != any_id) && (d_pre_id != sot_id))
                get_node(new_node->lxchild, before_id, pre_id, any_id, post_id, d_post_id);
            get_node(new_node->lchild, before_id, pre_id, NULL_ABC, post_id, d_post_id);
        }
        else
        {
            if ((pre_id != any_id) && (pre_id != sot_id))
                get_node(new_node->lxchild, before_id, any_id, NULL_ABC, post_id, d_post_id);
            get_node(new_node->lchild, before_id, NULL_ABC, NULL_ABC, post_id, d_post_id);
        }
    }
    if (post_id != NULL_ABC)
    {
        if (d_post_id != NULL_ABC)
        {
            if ((d_post_id != any_id) && (d_post_id != eot_id))
                get_node(new_node->rxchild, before_id, pre_id, d_pre_id, post_id, any_id);
            get_node(new_node->rchild, before_id, pre_id, d_pre_id, post_id, NULL_ABC);
        }
        else
        {
            if ((post_id != any_id) && (post_id != eot_id))
                get_node(new_node->rxchild, before_id, pre_id, d_pre_id, any_id, NULL_ABC);
            get_node(new_node->rchild, before_id, pre_id, d_pre_id, NULL_ABC, NULL_ABC);
        }
    }
}

void SiteSpace::get_nodes(Pool *tp, vec<vec<SiteNode *>> &outputs, const vec<vec<usi_t>> &sites)
{
    outputs.resize(sites.size());
    parallel_apply<true>(
        tp,
        [this](vec<SiteNode *> &outputs, const vec<usi_t> &sites) {
            outputs.resize(sites.size());
            for (size_t i = 0; i < sites.size(); i++)
                get_node(outputs[i], sites[i]);
        },
        outputs,
        sites);

    SPDLOG_TRACE("#sites {}", nodes.size());
}

GraphNode::GraphNode(SiteNode *base) : base(base) {}

GraphNode *SiteGraph::add_root(SiteNode *root, int order)
{
    GraphNode *gnode = generate_subgraph(root);
    vec<GraphNode *> gnodes = get_descendants(gnode);
    SPDLOG_DEBUG("Adding root, order {0} #nodes {1}.", order, gnodes.size());
    for (auto gnode : gnodes)
    {
        SPDLOG_TRACE("  Site: {}", site::str(gnode->base->site));
        gnode->num_sites++;
        gnode->linked_words.insert(order);
    }
    return gnode;
}

inline GraphNode *SiteGraph::generate_subgraph(SiteNode *snode)
{
    usi_t site = snode->site;
    if (nodes.find(site) == nodes.end())
    {
        GraphNode *gnode = new GraphNode(snode);
        nodes[site] = gnode;
        if (snode->lchild != nullptr)
            gnode->lchild = generate_subgraph(snode->lchild);
        if (snode->lxchild != nullptr)
            gnode->lxchild = generate_subgraph(snode->lxchild);
        if (snode->rchild != nullptr)
            gnode->rchild = generate_subgraph(snode->rchild);
        if (snode->rxchild != nullptr)
            gnode->rxchild = generate_subgraph(snode->rxchild);
        return gnode;
    }
    else
        return nodes[site];
}

inline void SiteGraph::visit(vec<GraphNode *> &queue, GraphNode *node)
{
    if ((node != nullptr) && (!node->visited))
    {
        node->visited = true;
        queue.push_back(node);
    }
}

vec<GraphNode *> SiteGraph::get_descendants(GraphNode *root)
{
    auto nodes = vec<GraphNode *>();
    visit(nodes, root);

    size_t i = 0;
    while (i < nodes.size())
    {
        GraphNode *node = nodes[i];
        visit(nodes, node->lchild);
        visit(nodes, node->lxchild);
        visit(nodes, node->rchild);
        visit(nodes, node->rxchild);
        i++;
    }
    for (const auto node : nodes)
        node->visited = false;
    return nodes;
}

SiteGraph::~SiteGraph()
{
    for (auto &item : nodes)
        delete item.second;
}
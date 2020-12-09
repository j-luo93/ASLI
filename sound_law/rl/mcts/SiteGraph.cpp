#include "SiteGraph.hpp"

GraphNode::GraphNode(SiteNode *base) : base(base) {}

SiteGraph::SiteGraph(SiteSpace *site_space) : site_space(site_space) {}

void SiteGraph::add_root(SiteNode *root, int order)
{
    GraphNode *g_node = generate_subgraph(root);
    std::vector<GraphNode *> g_nodes = get_descendants(g_node);
    for (auto g_node : g_nodes)
    {
        g_node->num_sites++;
        g_node->linked_words.insert(order);
    }
}

GraphNode *SiteGraph::generate_subgraph(SiteNode *s_node)
{
    usi_t site = s_node->site;
    if (nodes.find(site) == nodes.end())
    {
        GraphNode *g_node = new GraphNode(s_node);
        nodes[site] = g_node;
        if (s_node->lchild != nullptr)
            g_node->lchild = generate_subgraph(s_node->lchild);
        if (s_node->rchild != nullptr)
            g_node->rchild = generate_subgraph(s_node->rchild);
        return g_node;
    }
    else
        return nodes.at(site);
}

std::vector<GraphNode *> SiteGraph::get_descendants(GraphNode *root)
{
    std::vector<GraphNode *> nodes = std::vector<GraphNode *>{root};
    root->visited = true;
    size_t i = 0;
    while (i < nodes.size())
    {
        GraphNode *node = nodes.at(i);
        if ((node->lchild != nullptr) && (!node->lchild->visited))
        {
            nodes.push_back(node->lchild);
            node->lchild->visited = true;
        }
        if ((node->rchild != nullptr) && (!node->rchild->visited))
        {
            nodes.push_back(node->rchild);
            node->rchild->visited = true;
        }
        i++;
    }
    for (auto *node : nodes)
        node->visited = false;
    return nodes;
}

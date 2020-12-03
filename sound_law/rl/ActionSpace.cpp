#include <ActionSpace.h>

bool ActionSpace::use_conditional = false;

void ActionSpace::set_conditional(bool conditional)
{
    ActionSpace::use_conditional = conditional;
}

ActionSpace::ActionSpace()
{
    this->actions = vector<Action *>();
    this->word_cache = unordered_map<WordKey, Word *>();
    this->site_map = unordered_map<Site, vector<action_t>>();
    this->edges = unordered_map<abc_t, vector<abc_t>>();
    this->a2i_cache = vector<abc_t>();
}

void ActionSpace::register_edge(abc_t before_id, abc_t after_id)
{
    if (this->edges.find(before_id) == this->edges.end())
        this->edges[before_id] = vector<abc_t>();
    this->edges[before_id].push_back(after_id);
}

void ActionSpace::register_node(SiteNode *node)
{
    const Site &site = node->site;
    // Do not register duplicate sites.
    if (this->site_map.find(site) != this->site_map.end())
        return;

    this->site_map[site] = vector<action_t>();
    abc_t before_id, pre_id, d_pre_id, post_id, d_post_id;
    tie(before_id, pre_id, d_pre_id, post_id, d_post_id) = site;
    vector<abc_t> pre_cond = vector<abc_t>();
    if (pre_id != NULL_abc)
    {
        if (d_pre_id != NULL_abc)
            pre_cond.push_back(d_pre_id);
        pre_cond.push_back(pre_id);
    }
    vector<abc_t> post_cond = vector<abc_t>();
    if (post_id != NULL_abc)
    {
        post_cond.push_back(post_id);
        if (d_post_id != NULL_abc)
            post_cond.push_back(d_post_id);
    }
    for (abc_t after_id : this->edges[before_id])
    {
        action_t action_id = (action_t)this->actions.size();
        Action *action = new Action(action_id, before_id, after_id, pre_cond, post_cond);
        this->actions.push_back(action);
        this->a2i_cache.push_back(before_id);
        this->a2i_cache.push_back(after_id);
        this->a2i_cache.push_back(pre_id);
        this->a2i_cache.push_back(d_pre_id);
        this->a2i_cache.push_back(post_id);
        this->a2i_cache.push_back(d_post_id);
        this->site_map[site].push_back(action_id);
    }
    if (node->lchild != nullptr)
        this->register_node(node->lchild);
    if (node->rchild != nullptr)
        this->register_node(node->rchild);
}

Action *ActionSpace::get_action(action_t action_id)
{
    return this->actions.at(action_id);
}

inline bool is_more_retrictive(SiteNodeWithStats *node, SiteNodeWithStats *child)
{
    if (child == nullptr)
        return true;
    return (node->num_sites != child->num_sites);
}

void ActionSpace::set_action_allowed(TreeNode *node)
{
    if (!node->action_allowed.empty())
        return;
    // Build the graph first.
    const VocabIdSeq &vocab_i = node->vocab_i;
    SiteGraph graph = SiteGraph();
    for (size_t i = 0; i < vocab_i.size(); ++i)
    {
        const WordKey &key = get_word_key(vocab_i[i]);
        unique_lock<mutex> lock(this->mtx);
        if (this->word_cache.find(key) == this->word_cache.end())
        {
            Word *new_word = new Word(vocab_i[i], key);
            this->word_cache[key] = new_word;
            for (SiteNode *root : new_word->site_roots)
                this->register_node(root);
        }
        const Word *word = this->word_cache[key];
        lock.unlock();
        for (SiteNode *root : word->site_roots)
            graph.add_node(root);
    }

    vector<action_t> &action_allowed = node->action_allowed;
    for (auto const item : graph.nodes)
    {
        SiteNodeWithStats *snode = item.second;
        // If any child of this node has the same `num_sites`, then this node is discarded.
        if (is_more_retrictive(snode, snode->lchild) and is_more_retrictive(snode, snode->rchild))
        {
            unique_lock<mutex> lock(this->mtx);
            const vector<action_t> &map_values = this->site_map.at(snode->base->site);
            lock.unlock();
            action_allowed.insert(action_allowed.end(), map_values.begin(), map_values.end());
        }
    }
    assert(!action_allowed.empty());
}

size_t ActionSpace::size()
{
    return this->actions.size();
}

void ActionSpace::clear_cache()
{
    for (auto const &item : this->word_cache)
    {
        delete item.second;
    }
    this->word_cache.clear();
}

size_t ActionSpace::get_cache_size()
{
    return this->word_cache.size();
}

vector<abc_t> ActionSpace::expand_a2i()
{
    vector<abc_t> ret = this->a2i_cache;
    this->a2i_cache.clear();
    return ret;
}
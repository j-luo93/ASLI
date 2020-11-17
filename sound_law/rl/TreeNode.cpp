#include <TreeNode.h>
#include <Env.h>
#include <math.h>

long TreeNode::instance_cnt = 0;
mutex TreeNode::cls_mtx;
// Default values for edit distance computation.
vector<vector<long>> TreeNode::dist_mat = vector<vector<long>>();
long TreeNode::ins_cost = 1;
bool TreeNode::max_mode = false;

void common_init(TreeNode *node, const VocabIdSeq &vocab_i)
{
    node->vocab_i = vocab_i;
    node->played = false;
    {
        lock_guard<mutex> lock(TreeNode::cls_mtx);
        node->idx = TreeNode::instance_cnt;
        TreeNode::instance_cnt++;
    }
}

void TreeNode::set_dist_mat(vector<vector<long>> &dist_mat)
{
    TreeNode::dist_mat = dist_mat;
    TreeNode::ins_cost = 3;
}

void TreeNode::set_max_mode(bool max_mode)
{
    TreeNode::max_mode = max_mode;
}

TreeNode::TreeNode(const VocabIdSeq &vocab_i)
{
    // This constructor is used for end node only.
    common_init(this, vocab_i);
    this->end_node = nullptr;
    this->dist_to_end = 0;
    this->parent_node = nullptr;
    this->done = true;
    this->action_allowed = vector<long>();
};

TreeNode::TreeNode(const VocabIdSeq &vocab_i, TreeNode *end_node, const vector<long> &action_allowed)
{
    // This constructor is used for root node only.
    common_init(this, vocab_i);
    this->end_node = end_node;
    this->dist_to_end = node_distance(this, end_node, TreeNode::dist_mat, TreeNode::ins_cost);
    this->parent_node = nullptr;
    this->done = (this->vocab_i == end_node->vocab_i);
    this->action_allowed = action_allowed;
};

TreeNode::TreeNode(const VocabIdSeq &vocab_i, TreeNode *end_node, long best_i, long action_id, TreeNode *parent_node, const vector<long> &action_allowed)
{
    // This constructor is used for nodes created by one env step.
    common_init(this, vocab_i);
    this->end_node = end_node;
    this->dist_to_end = node_distance(this, end_node, TreeNode::dist_mat, TreeNode::ins_cost);
    this->prev_action = pair<long, long>(best_i, action_id);
    this->parent_node = parent_node;
    this->done = (this->vocab_i == end_node->vocab_i);
    this->action_allowed = action_allowed;
}

void TreeNode::add_edge(long action_id, const Edge &edge)
{
    // This will replace the old edge if it exists. Always call `has_acted` first.
    this->edges[action_id] = edge;
}

bool TreeNode::has_acted(long action_id)
{
    return this->edges.find(action_id) != this->edges.end();
}

long TreeNode::size()
{
    return this->vocab_i.size();
}

void TreeNode::lock()
{
    this->mtx.lock();
}

void TreeNode::unlock()
{
    this->mtx.unlock();
}

bool TreeNode::is_leaf()
{
    return this->prior.empty();
}

vector<float> TreeNode::get_scores(float puct_c)
{
    long sqrt_ns = sqrt(this->visit_count);
    vector<float> scores = vector<float>(this->prior.size());
    for (long i = 0; i < this->prior.size(); ++i)
    {
        long nsa = this->action_count[i];
        float q = this->total_value[i] / (nsa + 1e-8);
        float p = this->prior[i];
        float u = puct_c * p * sqrt_ns / (1 + nsa);
        scores[i] = q + u;
    }
    return scores;
}

long TreeNode::get_best_i(float puct_c)
{
    long best_i = -1;
    float best_v = NULL;
    vector<float> scores = this->get_scores(puct_c);
    for (long i = 0; i < this->prior.size(); ++i)
    {
        if ((best_v == NULL) or (scores[i] > best_v))
        {
            best_v = scores[i];
            best_i = i;
        };
    }
    assert(best_i != -1);
    return best_i;
}

void TreeNode::expand(const vector<float> &prior)
{
    this->prior = prior;
    long num_actions = this->action_allowed.size();
    assert(num_actions == prior.size());
    this->action_count = vector<long>(num_actions, 0);
    this->visit_count = 0;
    this->total_value = vector<float>(num_actions, 0.0);
    if (TreeNode::max_mode)
    {
        this->max_value = vector<float>(num_actions, -9999.9);
    }
}

void TreeNode::virtual_backup(long best_id, long game_count, float virtual_loss)
{
    this->action_count[best_id] += game_count;
    this->total_value[best_id] -= game_count * virtual_loss;
    this->visit_count += game_count;
}

void TreeNode::backup(float value, float mixing, long game_count, float virtual_loss)
{
    TreeNode *parent_node = this->parent_node;
    TreeNode *node = this;
    float rtg = 0.0;
    while ((parent_node != nullptr) and (!parent_node->played))
    {
        pair<long, long> pa = node->prev_action;
        long best_i = pa.first;
        long action_id = pa.second;
        float reward = parent_node->edges[action_id].second;
        rtg += reward;
        // float mixed_value = (1 - mixing) * value + mixing * reward;
        parent_node->action_count[best_i] -= game_count - 1;
        // parent_node->total_value[action_id] += game_count * virtual_loss + mixed_value;
        if (TreeNode::max_mode)
        {
            parent_node->max_value[best_i] = max(parent_node->max_value[best_i], value + rtg);
        }
        parent_node->total_value[best_i] += game_count * virtual_loss + value + rtg;
        parent_node->visit_count -= game_count - 1;
        node = parent_node;
        parent_node = node->parent_node;
    }
}

void TreeNode::reset()
{
    this->prior.clear();
    this->action_count.clear();
    this->visit_count = 0;
    this->total_value.clear();
    this->played = false;
    if (TreeNode::max_mode)
    {
        this->max_value.clear();
    }
}

void TreeNode::play()
{
    assert(!this->played);
    this->played = true;
}

list<pair<long, float>> TreeNode::get_path()
{
    TreeNode *node = this;
    list<pair<long, float>> path = list<pair<long, float>>();
    while (node->parent_node != nullptr)
    {
        long action_id = node->prev_action.second;
        Edge edge = node->parent_node->edges[action_id];
        float reward = edge.second;
        pair<long, float> path_edge = pair<long, float>(action_id, reward);
        path.push_front(path_edge);
        node = node->parent_node;
    }
    return path;
}

void TreeNode::clear_subtree()
{
    for (auto const &edge : this->edges)
    {
        TreeNode *node = edge.second.first;
        node->clear_subtree();
        delete edge.second.first;
    }
    this->edges.clear();
}

void TreeNode::add_noise(const vector<float> &noise, float noise_ratio)
{
    assert(!this->prior.empty());
    for (long i = 0; i < this->prior.size(); ++i)
    {
        this->prior[i] = this->prior[i] * (1.0 - noise_ratio) + noise[i] * noise_ratio;
    }
}

long TreeNode::get_num_allowed()
{
    return this->action_allowed.size();
}
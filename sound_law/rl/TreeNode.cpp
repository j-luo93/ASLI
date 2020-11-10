#include <TreeNode.h>
#include <Env.h>
#include <iostream>
#include <math.h>

long TreeNode::instance_cnt = 0;
mutex TreeNode::cls_mtx;

TreeNode::TreeNode(VocabIdSeq vocab_i)
{
    // This constructor is used for end node only.
    this->vocab_i = vocab_i;
    this->end_node = nullptr;
    this->dist_to_end = 0;
    this->prev_action = NULL;
    this->parent_node = nullptr;
    this->played = false;
    this->done = true;
    this->action_mask = vector<bool>();
    {
        lock_guard<mutex> lock(TreeNode::cls_mtx);
        this->idx = TreeNode::instance_cnt;
        TreeNode::instance_cnt++;
    }
};

TreeNode::TreeNode(VocabIdSeq vocab_i, TreeNode *end_node)
{
    // This constructor is used for root node only.
    this->vocab_i = vocab_i;
    this->end_node = end_node;
    this->dist_to_end = node_distance(this, end_node);
    this->prev_action = NULL;
    this->parent_node = nullptr;
    this->played = false;
    this->done = (this->vocab_i == end_node->vocab_i);
    this->action_mask = vector<bool>();
    {
        lock_guard<mutex> lock(TreeNode::cls_mtx);
        this->idx = TreeNode::instance_cnt;
        TreeNode::instance_cnt++;
    }
};

TreeNode::TreeNode(VocabIdSeq vocab_i, TreeNode *end_node, long action_id, TreeNode *parent_node)
{
    // This constructor is used for nodes created by one env step.
    this->vocab_i = vocab_i;
    this->end_node = end_node;
    this->dist_to_end = node_distance(this, end_node);
    this->prev_action = action_id;
    this->parent_node = parent_node;
    this->played = false;
    this->done = (this->vocab_i == end_node->vocab_i);
    this->action_mask = vector<bool>();
    {
        lock_guard<mutex> lock(TreeNode::cls_mtx);
        this->idx = TreeNode::instance_cnt;
        TreeNode::instance_cnt++;
    }
}

void TreeNode::add_edge(long action_id, Edge edge)
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
    // unique_lock<mutex> this->ulock(this->mtx);
    // FIXME(j_luo) how to use unique_lock?
    this->mtx.lock();
    // this->ulock = unique_lock<mutex> lock(this->mtx);
}

void TreeNode::unlock()
{
    // this->ulock.unlock();
    this->mtx.unlock();
}

bool TreeNode::is_leaf()
{
    return this->prior.empty();
}

long TreeNode::get_best_action_id(float puct_c)
{
    long sqrt_ns = sqrt(this->visit_count);
    long best_i = -1;
    float best_v = NULL;
    for (long i = 0; i < this->prior.size(); ++i)
    {
        if (not this->action_mask[i])
            continue;
        long nsa = this->action_count[i];
        float q = this->total_value[i] / (nsa + 1e-8);
        float p = this->prior[i];
        float u = puct_c * p * sqrt_ns / (1 + nsa);
        float score = q + u;
        if ((best_v == NULL) or (score > best_v))
        {
            best_v = score;
            best_i = i;
        };
    }
    assert(best_i != -1);
    return best_i;
}

void TreeNode::expand(vector<float> prior, vector<bool> action_mask)
{
    this->prior = prior;
    this->action_mask = action_mask;
    long num_actions = prior.size();
    this->action_count = vector<long>(num_actions, 0);
    this->visit_count = 0;
    this->total_value = vector<float>(num_actions, 0.0);
}

void TreeNode::virtual_backup(long action_id, long game_count, float virtual_loss)
{
    this->action_count[action_id] += game_count;
    this->total_value[action_id] -= game_count * virtual_loss;
    this->visit_count += game_count;
}

void TreeNode::backup(float value, float mixing, long game_count, float virtual_loss)
{
    TreeNode *parent_node = this->parent_node;
    TreeNode *node = this;
    while ((parent_node != nullptr) and (!parent_node->played))
    {
        long action_id = node->prev_action;
        float reward = parent_node->edges[action_id].second;
        float mixed_value = (1 - mixing) * value + mixing * reward;
        parent_node->action_count[action_id] -= game_count - 1;
        parent_node->total_value[action_id] += game_count * virtual_loss + mixed_value;
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
}

void TreeNode::unplay()
{
    // assert(this->played);
    this->played = false;
}

void TreeNode::play()
{
    assert(!this->played);
    this->played = true;
}
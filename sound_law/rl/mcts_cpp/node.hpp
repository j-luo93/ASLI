#pragma once

#include "common.hpp"
#include "word.hpp"

// This enum class documents which phase a node is in, in terms of finishing sampling an action.
enum class ActionPhase : int
{
    BEFORE,
    AFTER,
    PRE,
    D_PRE,
    POST,
    SPECIAL_TYPE,
};

using Affected = vec<pair<int, size_t>>;
using ChosenChar = pair<int, abc_t>;

class ActionSpace;
class Env;

/* ------------------------------------------------------------ */
/*                           Base Node                          */
/* ------------------------------------------------------------ */

// Base class for all nodes.
class BaseNode
{

    /* ----------------------- Edge-related ----------------------- */

private:
    friend class EdgeBuilder;
    friend class Traverser;

    vec<BaseNode *> children;
    // These two variables store the parents (potentially multiple) and the indices at which this node (as the child) is stored.
    vec<BaseNode *> parents;
    vec<size_t> parent_indices;

    // Whether this node has been visited. Used to traverse the graph.
    bool visited = false;

    // Disconnect the node from all of its parents.
    void disconnect_from_parents();
    // Disconnect the node from all of its children.
    void disconnect_from_children();
    // Connect `parent` and `child` at `index`.
    void connect(size_t index, BaseNode *child);
    // Initialize edge-related members (should be called after node expansion).
    void init_edges();

public:
    bool has_child(size_t) const;
    BaseNode *get_child(size_t) const;

    /* ---------------------- Memory-related ---------------------- */

private:
    friend class MemoryManager;

    // Whether this node is persistent, i.e., should not be destructued.
    bool persistent;

    // Make node persistent.
    void make_persistent();

public:
    bool is_persistent() const;

    /* ------------------ Multithreading-related ------------------ */
private:
    std::mutex mtx;

    /* ----------------------- Stats-related ---------------------- */

private:
    friend class StatsManager;

    vec<visit_t> action_counts;
    vec<float> total_values;
    vec<float> max_values;
    visit_t visit_count = 0;
    int max_index = -1;
    float max_value = -9999.9;

    void update_stats(size_t, float, int, float);
    void init_stats();
    void virtual_select(size_t, int, float);

public:
    const vec<visit_t> &get_action_counts() const;
    const vec<float> &get_total_values() const;
    visit_t get_visit_count() const;

    /* ---------------------- Action-related ---------------------- */

private:
    friend class ActionManager;

    void add_action(abc_t, const Affected &);
    void add_action(abc_t, Affected &&);
    void update_affected_at(size_t, int, size_t);
    void clear_priors();
    // Set prior to 0.0.
    void dummy_evaluate();

protected:
    vec<abc_t> permissible_chars; // What characters are permissible to act upon?
    vec<Affected> affected;       // What positions are affected by each permissible character?
    vec<float> priors;

public:
    const vec<abc_t> &get_actions() const;
    size_t get_num_actions() const;
    size_t get_action_index(abc_t) const;
    abc_t get_action_at(size_t) const;
    const Affected &get_affected_at(size_t) const;
    size_t get_num_affected_at(size_t) const;
    vec<float> get_scores(float, float, bool) const;
    // Given the current action phase, get the best action.
    ChosenChar get_best_action(float, float, bool) const;
    bool is_expanded() const;
    bool is_evaluated() const;
    // Play one mini-step.
    pair<BaseNode *, ChosenChar> play_mini() const;

    /* --------------------- Pruning-related --------------------- */

private:
    friend class PruningManager;

    int num_unpruned_actions = -1;
    vec<bool> pruned;

    void prune(size_t);
    void prune();
    bool is_pruned() const;
    void init_pruned();

public:
    const vec<bool> &get_pruned() const;

    /* -------------------------- Others -------------------------- */

protected:
    BaseNode(bool, bool);

    // Destructor is protected so that the derived classes can call it.
    ~BaseNode();

public:
    const bool stopped;

    virtual bool is_transitional() const = 0;
    virtual bool is_tree_node() const = 0;
};

/* ------------------------- Mini Node ------------------------ */

class TreeNode;
class MiniNode : public BaseNode
{
private:
    friend class NodeFactory;
    friend class ActionManager;

    void evaluate();

protected:
    MiniNode(const TreeNode *, ActionPhase, bool);

public:
    const TreeNode *base;
    const ActionPhase ap;

    bool is_tree_node() const override;

    virtual bool is_transitional() const;
};

/* ---------------------- Transition Node --------------------- */
// This is the last mini node that leads to a normal tree node. Only this node has rewards.

class TransitionNode : public MiniNode
{
private:
    friend class NodeFactory;
    friend class RewardManager;
    friend class ActionManager;

    TransitionNode(const TreeNode *, bool);

    vec<float> rewards;
    void init_rewards();
    void set_reward_at(size_t, float);

public:
    bool is_transitional() const override;
    float get_reward_at(size_t) const;
    const vec<float> &get_rewards() const;
};

/* ------------------------- Tree Node ------------------------ */

// FIXME(j_luo) weird
struct Subpath
{
    array<ChosenChar, 7> chosen_seq;
    array<MiniNode *, 6> mini_node_seq;
    bool stopped;
};

class TreeNode : public BaseNode
{

    /* -------------------- Constructor-related ------------------- */

private:
    friend class NodeFactory;
    friend class MemoryManager;

    void common_init(const vec<Word *> &);
    // This is used for persistent nodes (e.g., start and end nodes).
    TreeNode(const vec<Word *> &);
    // This is used for everything else.
    TreeNode(const vec<Word *> &, bool);

    // Static methods and members to manage construction.
    static Trie<Word *, TreeNode *> t_table;
    // Create a new node if it is not in the trie.
    static TreeNode *get_tree_node(const vec<Word *> &);
    static TreeNode *get_tree_node(const vec<Word *> &, bool);
    static void remove_node_from_t_table(TreeNode *);

public:
    static size_t get_num_nodes();

    /* -------------------------- Others -------------------------- */

private:
    friend class ActionManager;

    vec<vec<float>> meta_priors;
    vec<float> special_priors;
    float dist = 0.0;
    bool done = false;

    void evaluate(const vec<vec<float>> &, const vec<float> &);
    void add_noise(const vec<vec<float>> &, const vec<float> &, float);

public:
    // Given actions, evaluate their priors.
    vec<float> evaluate_actions(const vec<abc_t> &, ActionPhase) const;
    vec<float> evaluate_special_actions(const vec<abc_t> &) const;

    const vec<Word *> words;

    float get_dist() const;
    bool is_done() const;
    bool is_leaf() const;
    pair<TreeNode *, Subpath> play() const;
    const IdSeq &get_id_seq(int) const;
    size_t size() const;
    bool is_transitional() const override;
    bool is_tree_node() const override;
    pair<vec<vec<size_t>>, vec<vec<size_t>>> get_alignments() const;
};

namespace str
{
    inline string from(ActionPhase ap)
    {
        switch (ap)
        {
        case ActionPhase::BEFORE:
            return "BEFORE";
        case ActionPhase::AFTER:
            return "AFTER";
        case ActionPhase::PRE:
            return "PRE";
        case ActionPhase::D_PRE:
            return "D_PRE";
        case ActionPhase::POST:
            return "D_POST";
        case ActionPhase::SPECIAL_TYPE:
            return "SPECIAL_TYPE";
        }
    }

    inline string from(const BaseNode *node) { return "stopped: " + from(node->stopped); };

    inline string from(const TreeNode *node)
    {
        string out = from(static_cast<const BaseNode *>(node)) + "\n";
        for (const auto word : node->words)
        {
            for (const auto unit : word->id_seq)
                out += std::to_string(unit) + " ";
            out += "\n";
        }
        return out;
    }

    inline string from(const MiniNode *node) { return from(static_cast<const BaseNode *>(node)) + " phase: " + from(node->ap) + " base: " + from(node->base); }

} // namespace str

// Used to build edges.
class EdgeBuilder
{
    friend class ActionSpace;

    static void connect(BaseNode *parent, size_t index, BaseNode *child) { parent->connect(index, child); }
    static void init_edges(BaseNode *node) { node->init_edges(); }
};

// Used to traverse the graph.
class Traverser
{
    friend class ActionSpace;

    // Visit one node and append it to the queue if it hasn't been visited.
    static void visit(BaseNode *node, vec<BaseNode *> &queue)
    {
        if (!node->visited)
        {
            node->visited = true;
            queue.push_back(node);
        }
    };

    // Traverse from `start` using bfs.
    static vec<BaseNode *> bfs(BaseNode *start)
    {
        auto queue = vec<BaseNode *>();
        visit(start, queue);
        size_t i = 0;
        while (i < queue.size())
        {
            auto selected = queue[i];
            for (const auto child : selected->children)
                if (child != nullptr)
                    visit(child, queue);
            ++i;
        }

        for (const auto node : queue)
            node->visited = false;
        return queue;
    }
};

class LruCache;
class MemoryManager
{
    friend class LruCache;

    static void make_persistent(BaseNode *node) { node->make_persistent(); }
    // Release memory allocated to `node` by calling `delete`, and remove its entry in the `t_table` if needed.
    static void release(BaseNode *node)
    {
        if (node->is_tree_node())
            TreeNode::remove_node_from_t_table(static_cast<TreeNode *>(node));
        delete node;
    }
};

// Used by MCTS to update stats.
class Mcts;
class StatsManager
{
    friend class Mcts;

    static void update_stats(BaseNode *node, size_t index, float new_value, int game_count, float virtual_loss) { node->update_stats(index, new_value, game_count, virtual_loss); }
    static void virtual_select(BaseNode *node, size_t index, int game_count, float virtual_loss) { node->virtual_select(index, game_count, virtual_loss); }
};

// All useful methods invoked by ActionSpace, including initializing/evaluating nodes and action expansion.
class ActionManager
{
    friend class ActionSpace;

    static void add_action(BaseNode *node, abc_t action, const Affected &affected) { node->add_action(action, affected); }
    static void add_action(BaseNode *node, abc_t action, Affected &&affected) { node->add_action(action, affected); }
    static void update_affected_at(BaseNode *node, size_t index, int order, size_t pos) { node->update_affected_at(index, order, pos); }
    static void init_pruned(BaseNode *node) { node->init_pruned(); }
    static void init_stats(BaseNode *node) { node->init_stats(); };
    static void init_rewards(TransitionNode *node) { node->init_rewards(); }
    static void evaluate(TreeNode *node, const vec<vec<float>> &meta_priors, const vec<float> &special_priors) { node->evaluate(meta_priors, special_priors); }
    static void evaluate(MiniNode *node) { node->evaluate(); }
    static void add_noise(TreeNode *node, const vec<vec<float>> &meta_noise, const vec<float> &special_noise, float noise_ratio) { node->add_noise(meta_noise, special_noise, noise_ratio); }
    static void dummy_evaluate(BaseNode *node) { node->dummy_evaluate(); }
    static void clear_priors(BaseNode *node) { node->clear_priors(); }
};

class PruningManager
{
    friend class ActionSpace;
    friend class Mcts;

    static void prune(BaseNode *node, size_t index) { node->prune(index); }
};

class NodeFactory
{
    friend class ActionSpace;
    // FIXME(j_luo) env is rarely using it.
    friend class Env;

    static MiniNode *get_mini_node(const TreeNode *base, ActionPhase ap, bool stopped) { return new MiniNode(base, ap, stopped); }
    static TransitionNode *get_transition_node(const TreeNode *base, bool stopped) { return new TransitionNode(base, stopped); }
    static TreeNode *get_tree_node(const vec<Word *> &words) { return TreeNode::get_tree_node(words); }
    static TreeNode *get_tree_node(const vec<Word *> &words, bool stopped) { return TreeNode::get_tree_node(words, stopped); };
    static TreeNode *get_stopped_node(const TreeNode *node) { return new TreeNode(node->words, true); }
};

class RewardManager
{
    friend class Env;
    static void set_reward_at(TransitionNode *node, size_t index, float reward) { node->set_reward_at(index, reward); }
};
#pragma once

#include "common.hpp"
#include "word.hpp"

namespace node
{
    constexpr int END_DEPTH = -1;
}

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

/* ------------------------- Base Node ------------------------ */

class BaseNode
{
    friend class ActionSpace;

    // void connect(BaseNode *const, const ChosenChar &);

protected:
    std::mutex mtx;

    BaseNode(BaseNode *const, const ChosenChar &, bool);

    bool played = false;

    // Given the current action phase, get the best next mini node.
    ChosenChar get_best_subaction(float, int, float, float);
    void prune(int);
    virtual BaseNode *play();

public:
    virtual ~BaseNode() = default;

    const bool stopped;

    vec<abc_t> permissible_chars; // What characters are permissible to act upon?
    vec<Affected> affected;       // What positions are affected by each permissible character?
    vec<BaseNode *> children;

    vec<float> priors;
    vec<bool> pruned;
    vec<visit_t> action_counts;
    vec<float> total_values;
    vec<float> max_values;
    visit_t visit_count = 0;
    int max_index = -1;
    float max_value = -9999.9;
    int num_unpruned_actions = -1;

    bool is_expanded();
    bool is_evaluated();
    vec<float> get_scores(float, float);
    size_t get_num_actions();
    void prune();
    bool is_pruned();
    size_t get_num_descendants();

    virtual bool is_transitional() = 0;
    virtual bool is_tree_node() = 0;
};

/* ------------------------- Mini Node ------------------------ */

class TreeNode;

class MiniNode : public BaseNode
{
    friend class ActionSpace;
    friend class TreeNode;
    friend class TransitionNode;

    MiniNode(TreeNode *, BaseNode *const, const ChosenChar &, ActionPhase, bool);

public:
    virtual ~MiniNode() = default;

    TreeNode *const base;
    const ActionPhase ap;

    bool is_tree_node() override;

    virtual bool is_transitional();
};

/* ---------------------- Transition Node --------------------- */
// This is the last mini node that leads to a normal tree node. Only this node has rewards.

class TransitionNode : public MiniNode
{
    friend class ActionSpace;
    friend class Env;

    TransitionNode(TreeNode *, MiniNode *, const ChosenChar &, bool);

public:
    ~TransitionNode() override = default;

    vec<float> rewards;

    bool is_transitional() override;
};

/* ------------------------- Tree Node ------------------------ */

class Mcts;
class Env;

class TreeNode : public BaseNode
{
    friend class Mcts;
    friend class Env;
    friend class ActionSpace;

    static Trie<Word *, TreeNode *> t_table;
    static TreeNode *get_tree_node(const vec<Word *> &, int);
    static TreeNode *get_tree_node(const vec<Word *> &, int, BaseNode *const, const ChosenChar &, bool);

    vec<vec<float>> meta_priors;
    vec<float> special_priors;

    void common_init(const vec<Word *> &);
    TreeNode(const vec<Word *> &, int);
    TreeNode(const vec<Word *> &, int, BaseNode *const, const ChosenChar &, bool);

    TreeNode *play() override;

public:
    ~TreeNode() override = default;

    const vec<Word *> words;
    const int depth;

    float dist = 0.0;
    bool done = false;

    // Return a vector of `MiniNode *` as the subactions.
    bool is_leaf();
    IdSeq get_id_seq(int);
    size_t size();
    bool is_transitional() override;
    bool is_tree_node() override;
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

    inline string from(TreeNode *node)
    {
        string out = "";
        for (const auto word : node->words)
        {
            for (const auto unit : word->id_seq)
                out += std::to_string(unit) + " ";
            out += "\n";
        }
        return out;
    }
} // namespace str
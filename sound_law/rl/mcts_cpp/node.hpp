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
};

using Affected = vec<pair<int, size_t>>;
using ChosenChar = pair<int, abc_t>;

class ActionSpace;

/* ------------------------- Base Node ------------------------ */

class BaseNode
{
    friend class ActionSpace;

protected:
    std::mutex mtx;

    BaseNode(BaseNode *const, const ChosenChar &, bool);

    bool played = false;

    // Given the current action phase, get the best next mini node.
    ChosenChar get_best_subaction(float, int, float);
    void backup(float, int, float);

    virtual BaseNode *play();

public:
    BaseNode *const parent;
    const ChosenChar chosen_char;
    const bool stopped;

    vec<abc_t> permissible_chars; // What characters are permissible to act upon?
    vec<Affected> affected;       // What positions are affected by each permissible character?
    vec<BaseNode *> children;

    vec<float> priors;
    vec<visit_t> action_counts;
    vec<float> total_values;
    visit_t visit_count = 0;
    int max_index = -1;
    float max_value = -9999.9;

    bool is_expanded();
    bool is_evaluated();
    vec<float> get_scores(float);
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
    TreeNode *const base;
    const ActionPhase ap;
};

/* ---------------------- Transition Node --------------------- */
// This is the last mini node that leads to a normal tree node. Only this node has rewards.

class TransitionNode : public MiniNode
{
    friend class ActionSpace;
    friend class Env;

    TransitionNode(TreeNode *, MiniNode *, const ChosenChar &, bool);

public:
    vec<float> rewards;
};

/* ------------------------- Tree Node ------------------------ */

using MetaPriors = array<vec<float>, 6>;

class Mcts;
class Env;

class TreeNode : public BaseNode
{
    friend class Mcts;
    friend class Env;
    friend class ActionSpace;

    void common_init(const vec<Word *> &);
    TreeNode(const vec<Word *> &, int);
    TreeNode(const vec<Word *> &, int, BaseNode *const, const ChosenChar &, bool);

public:
    const vec<Word *> words;
    const int depth;

    float dist = 0.0;
    bool done = false;

    MetaPriors meta_priors;

    // Return a vector of `MiniNode *` as the subactions.
    bool is_leaf();
    TreeNode *play() override;
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
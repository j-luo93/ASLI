#include <chrono>

#include "site.hpp"
#include "word.hpp"
#include "action.hpp"
#include "tree_node.hpp"
#include "env.hpp"
#include "mcts.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "cxxopts.hpp"

inline float randf(float high)
{
    return high * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

inline float randint(int high)
{
    float ret = randf(static_cast<float>(high - 1));
    return static_cast<int>(ret);
}

VocabIdSeq randv(int num_words, int max_len, int num_abc)
{
    VocabIdSeq vocab = VocabIdSeq(num_words);
    for (int i = 0; i < num_words; i++)
    {
        vocab[i] = IdSeq(max_len);
        vocab[i][0] = 0;
        for (int j = 1; j < max_len - 1; j++)
            vocab[i][j] = randint(num_abc - 4) + 4;
        vocab[i][max_len - 1] = 1;
    }
    return vocab;
}

vec<float> randp(int num_actions)
{
    auto p = vec<float>(num_actions);
    float sum = 0.0;
    for (int i = 0; i < num_actions; i++)
    {
        p[i] = randf(1.0);
        sum += p[i];
    }
    for (int i = 0; i < num_actions; i++)
        p[i] = p[i] / sum;
    return p;
}

template <class T>
void add_argument(cxxopts::Options &options, const std::string &name, const std::string &desc, const std::string &default_v)
{
    options.add_options()(name, desc, cxxopts::value<T>()->default_value(default_v));
}

void add_flag(cxxopts::Options &options, const std::string &name, const std::string &desc)
{
    options.add_options()(name, desc);
}

int main(int argc, char *argv[])
{
    cxxopts::Options parser("test", "test program");
    add_argument<int>(parser, "num_threads", "Number of threads", "1");
    add_argument<int>(parser, "num_words", "Number of words", "10");
    add_argument<int>(parser, "max_len", "Max length", "10");
    add_argument<int>(parser, "num_abc", "Number of characters", "400");
    add_argument<int>(parser, "num_steps", "Number of steps", "20");
    add_argument<float>(parser, "puct_c", "puct constant", "5.0");
    add_argument<unsigned>(parser, "random_seed", "Random seed", "0");
    add_flag(parser, "log_to_file", "Flag to log to file");
    add_flag(parser, "quiet", "Set log level to error to disable info logging.");
    auto args = parser.parse(argc, argv);
    const int num_threads = args["num_threads"].as<int>();
    const int num_words = args["num_words"].as<int>();
    const int max_len = args["max_len"].as<int>();
    const int num_abc = args["num_abc"].as<int>();
    const int num_steps = args["num_steps"].as<int>();
    const float puct_c = args["puct_c"].as<float>();
    const unsigned random_seed = args["random_seed"].as<unsigned>();
    const bool log_to_file = args["log_to_file"].as<bool>();
    const bool quiet = args["quiet"].as<bool>();

    srand(random_seed);
    std::cerr << "num threads " << num_threads << '\n';

    spdlog::level::level_enum level;
    if (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE)
        level = spdlog::level::trace;
    else if (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG)
        level = spdlog::level::debug;
    if (quiet)
        level = spdlog::level::err;
    spdlog::set_level(level);
    if (log_to_file)
    {
        auto logger = spdlog::basic_logger_mt("default", "log.txt", true);
        spdlog::set_default_logger(logger);
        logger->flush_on(level);
    }

    Pool *tp = nullptr;
    if (num_threads > 1)
        tp = new Pool(num_threads);

    vec<vec<float>> dist_mat = vec<vec<float>>();
    float ins_cost = static_cast<float>(num_abc);

    for (int i = 0; i < num_abc; i++)
    {
        dist_mat.push_back(vec<float>(num_abc));
        for (int j = 0; j < num_abc; j++)
            dist_mat[i][j] = std::abs(i - j);
    }

    VocabIdSeq start_ids = randv(num_words, max_len, num_abc);
    VocabIdSeq end_ids = randv(num_words, max_len, num_abc);

    auto site_space = new SiteSpace(0, 1, 2, 3);
    auto word_space = new WordSpace(site_space, dist_mat, ins_cost);
    auto action_space = new ActionSpace(site_space, word_space, 0.0, 1);
    auto env = new Env(action_space, word_space, start_ids, end_ids, 1.0, 0.02);
    for (int i = 4; i < num_abc; i++)
        for (int j = std::max(0, i - 10); j < std::min(num_abc, i + 11); j++)
            if ((i != j) && (j > 3))
                action_space->register_edge(i, j);
    action_space->set_action_allowed(tp, vec<TreeNode *>{env->start});
    env->start->expand(randp(env->start->action_allowed.size()));

    show_size(*site_space, "site space");
    show_size(*word_space, "word space");

    auto mcts = new Mcts(env, puct_c, 3, 0.5, num_threads);
    int num_sims = 1000;
    int batch_size = 40;
    TreeNode *root = env->start;
    SPDLOG_INFO("Start:\n{}", env->start->str());
    SPDLOG_INFO("End:\n{}", env->end->str());
    action_space->timer.disable();
    for (int i = 0; i < num_steps; i++)
    {
        if (i == num_steps / 2)
            action_space->timer.enable();
        if ((root->stopped) || (root->done))
            break;
        SPDLOG_INFO("Step: {}", i + 1);
        SPDLOG_DEBUG("Current root:\n{}", root->str());
        show_size(root->action_allowed, "#actions");
        for (int j = 0; j < num_sims / batch_size; j++)
        {
            auto selected = mcts->select(root, batch_size, num_steps);
            auto unique_nodes = vec<TreeNode *>();
            find_unique(unique_nodes, selected,
                        [](TreeNode *tnode) {
                            return ((!tnode->done) && (!tnode->stopped));
                        });
            SPDLOG_DEBUG("#nodes to expand: {}", unique_nodes.size());
            for (const auto node : unique_nodes)
                node->expand(randp(node->action_allowed.size()));
            mcts->backup(selected, vec<float>(selected.size(), 0.0));
        }
        auto action_to_play = mcts->play(root);
        root = root->neighbors[action_to_play];
    }
    action_space->timer.show_stats();
    std::cerr << site_space->size() << " sites explored\n";
    std::cerr << word_space->size() << " words explored\n";
}

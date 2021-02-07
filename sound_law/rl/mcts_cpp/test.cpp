#include <chrono>

#include "word.hpp"
#include "action.hpp"
#include "node.hpp"
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
    add_argument<int>(parser, "num_sims", "Number of simulations", "1000");
    add_argument<int>(parser, "batch_size", "Batch size per evaluation", "40");
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
    const int num_sims = args["num_sims"].as<int>();
    const int batch_size = args["batch_size"].as<int>();

    srand(random_seed);
    std::cerr << "num threads " << num_threads << '\n';

    spdlog::level::level_enum level = spdlog::level::info;
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

    auto env_opt = EnvOpt();
    env_opt.start_ids = start_ids;
    env_opt.end_ids = end_ids;
    env_opt.final_reward = 1.0;
    env_opt.step_penalty = 0.001;
    auto ws_opt = WordSpaceOpt();
    ws_opt.dist_mat = dist_mat;
    ws_opt.ins_cost = ins_cost;
    auto env = new Env(env_opt, ws_opt);
    for (int i = 4; i < num_abc; i++)
        for (int j = std::max(0, i - 10); j < std::min(num_abc, i + 11); j++)
            if ((i != j) && (j > 3))
                env->action_space->register_permissible_change(i, j);
    env->action_space->evaluate(env->start,
                                MetaPriors{
                                    randp(num_abc), randp(num_abc), randp(num_abc), randp(num_abc), randp(num_abc), randp(num_abc)});

    auto mcts_opt = MctsOpt();
    mcts_opt.puct_c = puct_c;
    mcts_opt.game_count = 3;
    mcts_opt.virtual_loss = 0.5;
    mcts_opt.num_threads = num_threads;
    auto mcts = new Mcts(env, mcts_opt);
    TreeNode *root = env->start;
    // SPDLOG_INFO("Start:\n{}", env->start->str());
    // SPDLOG_INFO("End:\n{}", env->end->str());
    // action_space->timer.disable();
    SPDLOG_INFO("Start dist: {}", root->dist);
    SPDLOG_INFO("Start node str:\n{}", str::from(env->start));
    SPDLOG_INFO("End node str:\n{}", str::from(env->end));
    for (int i = 0; i < num_steps; i++)
    {
        // if (i == num_steps / 2)
        //     action_space->timer.enable();
        if ((root->stopped) || (root->done))
            break;
        SPDLOG_INFO("Step: {}", i + 1);
        // SPDLOG_DEBUG("Current root:\n{}", root->str());
        show_size(root->permissible_chars, "#actions");
        for (int j = 0; j < num_sims / batch_size; j++)
        {
            auto selected = mcts->select(root, batch_size, num_steps);
            auto unique_nodes = vec<TreeNode *>();
            unique_nodes = find_unique(selected,
                                       [](TreeNode *node) {
                                           return ((!node->done) && (!node->stopped));
                                       });
            SPDLOG_DEBUG("#nodes to evaluate: {}", unique_nodes.size());
            for (const auto node : unique_nodes)
                env->action_space->evaluate(node,
                                            MetaPriors{
                                                randp(num_abc), randp(num_abc), randp(num_abc), randp(num_abc), randp(num_abc), randp(num_abc)});
            SPDLOG_DEBUG("Backing up values.");
            mcts->backup(selected, vec<float>(selected.size(), 0.0));
        }
        // auto scores = root->get_scores(puct_c);
        // for (size_t i = 0; i < root->permissible_chars.size(); ++i)
        //     std::cerr << root->permissible_chars[i] << ":" << scores[i] << " ";
        // std::cerr << "\n";
        // std::cerr << "max index: " << root->max_index << " max_value: " << root->max_value << "\n";
        root = root->play();
        SPDLOG_INFO("New dist: {}", root->dist);
    }
    // action_space->timer.show_stats();
    // std::cerr << site_space->size() << " sites explored\n";
    // std::cerr << word_space->size() << " words explored\n";
}

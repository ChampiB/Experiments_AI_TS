//
// Created by Theophile Champion on 01/07/2021.
//

#include "environments/GraphEnv.h"
#include "environments/MazeEnv.h"
#include "environments/EnvType.h"
#include "algorithms/planning/MCTSConfig.h"
#include "trackers/TimeTracker.h"
#include "trackers/GraphPerformanceTracker.h"
#include "trackers/MazePerformanceTracker.h"
#include "zoo/BTAI.h"
#include "api/API.h"
#include "math/Ops.h"
#include <iostream>
#include <algorithms/planning/EvaluationType.h>

using namespace hopi::environments;
using namespace hopi::graphs;
using namespace hopi::zoo;
using namespace hopi::api;
using namespace hopi::math;
using namespace hopi::algorithms::planning;
using namespace experiments;
using namespace experiments::trackers;
using namespace torch;

torch::Tensor advancedPrefMaze1() {
    Tensor D_tilde = API::full({22}, 0.0016666666);
    std::vector<int> good_states{0,1,2,3,4,7,12,15,21};

    D_tilde[5] = 0.8;
    for (int good_state : good_states) {
        D_tilde[good_state] = 0.02;
    }
    return D_tilde;
}

torch::Tensor advancedPrefMaze5() {
    Tensor D_tilde = API::full({7}, 0.0333333);
    std::vector<int> good_states{1,3,5};

    D_tilde[2] = 0.45;
    for (int good_state : good_states) {
        D_tilde[good_state] = 0.15;
    }
    return D_tilde;
}

std::vector<std::pair<int, int>> getLocalMinima(const std::string &key) {
    static std::map<std::string, std::vector<std::pair<int, int>>> map = {
            {"1.maze", {{3,4}}},
            {"5.maze", {{3,3}}},
            {"7.maze", {{8,3}, {4,7}, {1,3}, {4,1}, {6,3}, {4,5}}},
            {"8.maze", {{3,7}, {7,7}}},
            {"9.maze", {{5,3},{3,5}}},
            {"14.maze", {{3,4}, {7,4}}},
            {"15.maze", {{3,8}}}
    };
    return map[key];
}

int main() {
    // Demo hyper-parameters.
    int NB_SIMULATIONS = 100;
    int NB_ACTION_PERCEPTION_CYCLES = 20;
    std::string OUTPUT_FILE_NAME = "../results/btai_bf_results.txt";

    // Maze environment hyper-parameters
    EnvType envType = EnvType::GRAPH;

    // Maze environment hyper-parameters
    std::string MAZES_PATH = "../Homing-Pigeon/examples/mazes/";
    std::string MAZE_FILE_NAME = "5.maze";
    std::string FULL_MAZE_FILE_NAME = MAZES_PATH + MAZE_FILE_NAME;
    std::vector<std::pair<int,int>> LOCAL_MINIMA = getLocalMinima(MAZE_FILE_NAME);

    // Graph environment hyper-parameters
    int NB_GOOD_PATHS = 3;
    int NB_BAD_PATHS = 5;
    std::vector<int> GOOD_PATHS_SIZES = {6,5,8};
    // MCTS hyper-parameters
    int    NB_PLANNING_STEPS = 25;
    double EXPLORATION_CONSTANT = 2;
    double PRECISION_PRIOR_PREFERENCES = 3;
    double PRECISION_ACTION_SELECTION = 100;
    EvaluationType EVALUATION_TYPE = EvaluationType::EFE;

    // Open the file in which the result should be written.
    std::ofstream file;
    file.open(OUTPUT_FILE_NAME, std::ios_base::app);
    file << "========== EXPERIMENT CONFIGURATION ==========" << std::endl;
    file << "NB_SIMULATIONS: " << NB_SIMULATIONS << std::endl;
    file << "NB_ACTION_PERCEPTION_CYCLES: " << NB_ACTION_PERCEPTION_CYCLES << std::endl;
    if (envType == EnvType::GRAPH) {
        file << "NB_GOOD_PATHS: " << NB_GOOD_PATHS << std::endl;
        file << "NB_BAD_PATHS: " << NB_BAD_PATHS << std::endl;
        file << "GOOD_PATHS_SIZES: " << GOOD_PATHS_SIZES << std::endl;
    } else {
        file << "MAZE_FILE_NAME: " << MAZE_FILE_NAME << std::endl;
        file << "LOCAL_MINIMA: " << LOCAL_MINIMA << std::endl;
    }
    file << "NB_PLANNING_STEPS: " << NB_PLANNING_STEPS << std::endl;
    file << "EXPLORATION_CONSTANT: " << EXPLORATION_CONSTANT << std::endl;
    file << "PRECISION_PRIOR_PREFERENCES: " << PRECISION_PRIOR_PREFERENCES << std::endl;
    file << "PRECISION_ACTION_SELECTION: " << PRECISION_ACTION_SELECTION << std::endl;
    file << "EVALUATION_TYPE: " << EVALUATION_TYPE << std::endl << std::endl;

    // Create environment.
    std::shared_ptr<Environment> env;
    if (envType == EnvType::GRAPH)
        env = GraphEnv::create(NB_GOOD_PATHS, NB_BAD_PATHS, GOOD_PATHS_SIZES);
    else
        env = MazeEnv::create(FULL_MAZE_FILE_NAME);

    // Create prior preferences.
    Tensor OBS_PREFERENCES = env->observations() - API::range(0, env->observations());
    Tensor STATES_PREFERENCES = Ops::uniform({env->states()}); // or advancedPrefMaze5()

    // Create MCTS configuration.
    auto tConfig = MCTSConfig::create(
            OBS_PREFERENCES,
            STATES_PREFERENCES,
            NB_PLANNING_STEPS,
            EXPLORATION_CONSTANT,
            PRECISION_PRIOR_PREFERENCES,
            PRECISION_ACTION_SELECTION
    );

    // Create time and performance trackers.
    std::unique_ptr<PerformanceTracker> perf_tracker;
    if (envType == EnvType::GRAPH)
        perf_tracker = GraphPerformanceTracker::create();
    else
        perf_tracker = MazePerformanceTracker::create(LOCAL_MINIMA);
    auto time_tracker = TimeTracker::create();

    // Initialise trackers.
    perf_tracker->reset();
    time_tracker->tic();

    // Run the episodes.
    for (int j = 0; j < NB_SIMULATIONS; ++j) {

        // Reset environment and create agent.
        auto obs = env->reset();
        auto agent = BTAI::create(env.get(), tConfig, obs);

        // Run one episode.
        for (int k = 0; k < NB_ACTION_PERCEPTION_CYCLES; ++k) {
            agent->step(env, EVALUATION_TYPE);
            if (env->solved()) {
                break;
            }
        }

        // Evaluate simulation.
        perf_tracker->track(env);
    }

    // Print trackers results
    time_tracker->toc();
    time_tracker->print(file);
    perf_tracker->print(file);

    return EXIT_SUCCESS;
}


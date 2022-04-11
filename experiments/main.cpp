//
// Created by Theophile Champion on 01/07/2021.
//

#include <graphs/FactorGraph.h>
#include <environments/GraphEnv.h>
#include <environments/MazeEnv.h>
#include <environments/EnvType.h>
#include <algorithms/planning/MCTSConfig.h>
#include <trackers/TimeTracker.h>
#include <trackers/GraphPerformanceTracker.h>
#include <trackers/MazePerformanceTracker.h>
#include <zoo/BTAI.h>
#include <environments/FrozenLakeEnv.h>
#include <iostream>
#include <algorithms/planning/EvaluationType.h>
#include <trackers/FrozenLakePerformanceTracker.h>
#include <environments/DisentangleSpritesEnv.h>
#include <trackers/SpritesPerformanceTracker.h>

using namespace hopi::environments;
using namespace hopi::graphs;
using namespace hopi::zoo;
using namespace hopi::algorithms::planning;
using namespace experiments;
using namespace experiments::trackers;
using namespace torch;
using namespace std;

// Define a type representing a pair containing an environment and a performance tracker.
typedef pair<shared_ptr<Environment>, unique_ptr<PerformanceTracker>> EPT_Pair;

/**
 * This function returns the position of the local minimum of the maze whose name is passed as parameters.
 * @param key the name of maze.
 * @return a vector containing the local minimum positions.
 */
vector<pair<int, int>> getLocalMinima(const std::string &key) {
    static map<string, vector<pair<int, int>>> map {
            {"1.maze", {{3,4}}},
            {"5.maze", {{3,3}}},
            {"9.maze", {{3,5},{5,3}}}
    };
    return map[key];
}

/**
 * This function returns the environment and performance tracker for the maze environment.
 * @param file in which the loaded environment should be logged.
 * @return a pair containing the environment and the performance tracker.
 */
EPT_Pair getMazeEnvAndPerfTracker(ofstream &file) {
    // Maze environment hyper-parameters
    string MAZES_PATH = "../Homing-Pigeon/examples/mazes/";
    string MAZE_FILE_NAME = "9.maze";
    string FULL_MAZE_FILE_NAME = MAZES_PATH + MAZE_FILE_NAME;
    vector<pair<int,int>> LOCAL_MINIMA = getLocalMinima(MAZE_FILE_NAME);

    // Create the environment and performance tracker.
    shared_ptr<Environment> env = MazeEnv::create(FULL_MAZE_FILE_NAME);
    unique_ptr<PerformanceTracker> perf_tracker = MazePerformanceTracker::create(LOCAL_MINIMA);

    // Log the loaded environment.
    file << "MAZE_FILE_NAME: " << MAZE_FILE_NAME << endl;
    file << "LOCAL_MINIMA: " << LOCAL_MINIMA << endl;

    return make_pair<>(env, move(perf_tracker));
}

/**
 * This function returns the environment and performance tracker for the graph environment.
 * @param file in which the loaded environment should be logged.
 * @return a pair containing the environment and the performance tracker.
 */
EPT_Pair getGraphEnvAndPerfTracker(ofstream &file) {
    // Hyper-parameters of the graph environment.
    int NB_GOOD_PATHS = 3;
    int NB_BAD_PATHS = 5;
    vector<int> GOOD_PATHS_SIZES = {6,5,8};

    // Create the environment and performance tracker.
    shared_ptr<Environment> env = GraphEnv::create(NB_GOOD_PATHS, NB_BAD_PATHS, GOOD_PATHS_SIZES);
    unique_ptr<PerformanceTracker> perf_tracker = GraphPerformanceTracker::create();

    // Log the loaded environment.
    file << "NB_GOOD_PATHS: " << NB_GOOD_PATHS << endl;
    file << "NB_BAD_PATHS: " << NB_BAD_PATHS << endl;
    file << "GOOD_PATHS_SIZES: " << GOOD_PATHS_SIZES << endl;

    return make_pair<>(env, move(perf_tracker));
}

/**
 * This function returns the environment and performance tracker for the frozen lake environment.
 * @param file in which the loaded environment should be logged.
 * @return a pair containing the environment and the performance tracker.
 */
EPT_Pair getFrozenLakeEnvAndPerfTracker(ofstream &file) {
    // Hyper-parameters of the frozen lake environment.
    string LAKES_PATH = "../Homing-Pigeon/examples/lakes/";
    string LAKE_FILE_NAME = "5.lake";
    string FULL_LAKE_FILE_NAME = LAKES_PATH + LAKE_FILE_NAME;

    // Create the environment and performance tracker.
    shared_ptr<Environment> env = FrozenLakeEnv::create(FULL_LAKE_FILE_NAME);
    unique_ptr<PerformanceTracker> perf_tracker = FrozenLakePerformanceTracker::create();

    // Log the loaded environment.
    file << "LAKE_FILE_NAME: " << LAKE_FILE_NAME << endl;

    return make_pair<>(env, move(perf_tracker));
}

/**
 * This function returns the environment and performance tracker for the dSprites environment.
 * @param file in which the loaded environment should be logged.
 * @return a pair containing the environment and the performance tracker.
 */
EPT_Pair getSpritesEnvAndPerfTracker(ofstream &file) {
    // Hyper-parameters of the d-sprites environment.
    string D_SPRITES_PATH = "../Homing-Pigeon/examples/d_sprites/";

    int GRANULARITY = 4; // Granularity of x and y position, i.e., 1, 2, 4, or 8.
                         // Granularity of 1 => agent see each position
                         // Granularity of 2 => agent see each 2x2 square as a single position
                         // ...
    int REPEAT = 8; // The number of times an action is repeated before the next perception cycle.

    // Create the environment and performance tracker.
    shared_ptr<Environment> env = DisentangleSpritesEnv::create(D_SPRITES_PATH, GRANULARITY, REPEAT);
    unique_ptr<PerformanceTracker> perf_tracker = SpritesPerformanceTracker::create();

    // Log the loaded environment.
    file << "D_SPRITES_FILE_NAME: " << D_SPRITES_PATH << endl;
    file << "GRANULARITY: " << GRANULARITY << endl;
    file << "REPEAT: " << REPEAT << endl;

    return make_pair<>(env, move(perf_tracker));
}

/**
 * This function returns the environment in which the agent should be run, and the associated tracker of
 * performance.
 * @param type the type of environment.
 * @param file in which the loaded environment should be described.
 * @return a pair containing the environment and the performance tracker.
 */
EPT_Pair getEnvAndPerfTracker(EnvType type, ofstream &file) {
    static map<EnvType, EPT_Pair (*)(ofstream &file)> map {
            {EnvType::MAZE,        &getMazeEnvAndPerfTracker},
            {EnvType::GRAPH,       &getGraphEnvAndPerfTracker},
            {EnvType::FROZEN_LAKE, &getFrozenLakeEnvAndPerfTracker},
            {EnvType::D_SPRITES,   &getSpritesEnvAndPerfTracker}
    };

    file << "========== ENVIRONMENT CONFIGURATION ==========" << std::endl;
    auto [env, perf_tracker] = (*map[type])(file);
    file << endl;
    return make_pair<>(env, move(perf_tracker));
}

/**
 * This function transform the environment name (string) into the environment type (EnvType).
 * @param name the environment name.
 * @return the environment type.
 */
EnvType getEnvType(char *name) {
    static map<string, EnvType> map {
            {"maze",    EnvType::MAZE},
            {"graph",   EnvType::GRAPH},
            {"lake",    EnvType::FROZEN_LAKE},
            {"sprites", EnvType::D_SPRITES}
    };

    return map[string(name)];
}

int main(int argc, char *argv[]) {

    // Open the file in which the result should be written.
    ofstream file;
    file.open("../results/BTAI_BF_section_3.txt", std::ios_base::app);

    // Get environment type.
    EnvType envType = (argc <= 1) ? EnvType::GRAPH : getEnvType(argv[1]);

    // Get environment and performance tracker.
    auto [env, perf_tracker] = getEnvAndPerfTracker(envType, file);

    // Create prior preferences.
    Tensor OBS_PREF = env->pref_obs();
    Tensor STATES_PREF = env->pref_states(false);

    // Demo hyper-parameters.
    int NB_SIMULATIONS = 100;
    int NB_ACTION_PERCEPTION_CYCLES = 20;

    // BTAI hyper-parameters
    int    NB_PLANNING_STEPS = 100;
    double EXPLORATION_CONSTANT = 2;
    double PRECISION_PRIOR_PREFERENCES = 3;
    double PRECISION_ACTION_SELECTION = 100;
    EvaluationType EVALUATION_TYPE = EvaluationType::EFE;

    // Create MCTS configuration.
    auto tConfig = MCTSConfig::create(
            OBS_PREF,
            STATES_PREF,
            NB_PLANNING_STEPS,
            EXPLORATION_CONSTANT,
            PRECISION_PRIOR_PREFERENCES,
            PRECISION_ACTION_SELECTION
    );

    // Create time tracker.
    auto time_tracker = TimeTracker::create(NB_SIMULATIONS);

    // Log the experiment configuration.
    file << "========== EXPERIMENT CONFIGURATION ==========" << std::endl;
    file << "NB_SIMULATIONS: " << NB_SIMULATIONS << std::endl;
    file << "NB_ACTION_PERCEPTION_CYCLES: " << NB_ACTION_PERCEPTION_CYCLES << std::endl;
    file << "NB_PLANNING_STEPS: " << NB_PLANNING_STEPS << std::endl;
    file << "EXPLORATION_CONSTANT: " << EXPLORATION_CONSTANT << std::endl;
    file << "PRECISION_PRIOR_PREFERENCES: " << PRECISION_PRIOR_PREFERENCES << std::endl;
    file << "PRECISION_ACTION_SELECTION: " << PRECISION_ACTION_SELECTION << std::endl;
    file << "EVALUATION_TYPE: " << EVALUATION_TYPE << std::endl << std::endl;

    // Initialise trackers.
    perf_tracker->reset();

    // Run the episodes.
    for (int j = 0; j < NB_SIMULATIONS; ++j) {

        // Reset environment and create agent.
        auto obs = env->reset();
        auto agent = BTAI::create(env.get(), tConfig, obs);

        // Run one episode.
        // env->print(); //TODO
        time_tracker->tic();
        for (int k = 0; k < NB_ACTION_PERCEPTION_CYCLES; ++k) {
            agent->step(env, EVALUATION_TYPE);
            // env->print(); //TODO
            if (env->solved())
                break;
        }
        time_tracker->toc();

        // Clean up memory.
        FactorGraph::setCurrent(nullptr);

        // Evaluate simulation.
        perf_tracker->track(env);
    }

    // Print trackers results
    perf_tracker->print(file);
    time_tracker->print(file);

    return EXIT_SUCCESS;
}

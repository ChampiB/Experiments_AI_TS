//
// Created by Theophile Champion on 15/12/2020.
//

#include "distributions/Transition.h"
#include "environments/Environment.h"
#include "environments/MazeEnv.h"
#include "nodes/VarNode.h"
#include "graphs/FactorGraph.h"
#include "algorithms/AlgoTree.h"
#include "algorithms/AlgoVMP.h"
#include "math/Ops.h"
#include "api/API.h"
#include <iostream>
#include <chrono>

using namespace hopi::environments;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
using namespace hopi::api;
using namespace hopi::algorithms;
using namespace torch;

void run_simulation(MazeEnv *env, int nb_AP_steps, int nb_P_steps) {
    /**
     ** Delete previous factor graph if any.
     **/
    FactorGraph::setCurrent(nullptr);

    /**
     ** Create the model's parameters.
     **/
    Tensor U0 = Ops::uniform({env->actions()});
    Tensor A  = env->A();
    Tensor B = env->B();
    Tensor D0 = env->D();

    /**
     ** Create the generative model.
     **/
    VarNode *a0 = API::Categorical(U0);
    VarNode *s0 = API::Categorical(D0);
    VarNode *o0 = API::Transition(s0, A);
    o0->setType(VarNodeType::OBSERVED);
    o0->setName("o0");
    VarNode *s1 = API::ActiveTransition(s0, a0, B);
    VarNode *o1 = API::Transition(s1, A);
    o1->setName("o1");
    o1->setType(VarNodeType::OBSERVED);
    auto fg = FactorGraph::current();
    fg->setTreeRoot(s1);
    fg->loadEvidence(env->observations(), "../Homing-Pigeon/examples/evidences/1.evi");

    /**
     ** Create the model's prior preferences.
     **/
    // Advanced prior preferences over hidden state:
    // D_tilde(5,0) = 0.8;
    // std::vector<int> good_states{0,1,2,3,4,7,12,15,21};
    // for (int good_state : good_states) {
    //     D_tilde(good_state,0) = 0.02;
    // }

    // Simple prior preferences over hidden state:
    Tensor D_tilde = Ops::uniform({env->states()});

    // Prior preferences over observation:
    Tensor E_tilde = softmax(env->observations() - API::range(0, env->observations()), 0);

    /**
     ** Run the simulation.
     **/
    for (int i = 0; i < nb_AP_steps; ++i) { // Action perception cycle
        AlgoVMP::inference(fg->getNodes());
        auto algoTree = std::make_unique<AlgoTree>(env->actions(), D_tilde, E_tilde);
        for (int j = 0; j < nb_P_steps; ++j) { // Planning
            VarNode *n = algoTree->nodeSelection(fg);
            algoTree->expansion(n, A, B);
            AlgoVMP::inference(algoTree->lastExpandedNodes());
            algoTree->evaluation();
            algoTree->propagation(n, fg->treeRoot());
        }
        env->print();
        int a = algoTree->actionSelection(fg->treeRoot());
        int o = env->execute(a);
        fg->integrate(a, Ops::one_hot(env->observations(), o), A, B);
    }
}

int main()
{
    // Number of experiments
    int E = 1;
    // Number of action perception cycles
    int AP = 15;
    // Number of planning steps
    int P  = 100;
    // Number of simulations
    int N  = 100;

    std::cout << std::endl;
    std::cout << "Number of experiments: " << E << std::endl;
    std::cout << "Number of action perception cycles: " << AP << std::endl;
    std::cout << "Number of planning iterations: " << P << std::endl;
    std::cout << "Number of simulations: " << N << std::endl;
    std::cout << std::endl;
    std::cout << "P(global),P(local),h,m,s,ms " << std::endl;

    for (int j = 0; j < E; ++j) { // Run the experiment E times

        double global_min = 0;
        double local_min = 0;

        auto begin = std::chrono::steady_clock::now();

        for (int i = 0; i < N; ++i) { // For N simulations
            auto env = std::make_unique<MazeEnv>("../Homing-Pigeon/examples/mazes/1.maze");
            run_simulation(env.get(), AP, P);
            auto exit_pos = env->exitPosition();
            auto local_1_pos = std::make_pair(3,4);
            auto agent_pos = env->agentPosition();
            if (env->manhattan_distance(agent_pos, exit_pos) < env->manhattan_distance(agent_pos, local_1_pos)) {
                global_min += 1;
            } else {
                local_min += 1;
            }
        }

        auto end = std::chrono::steady_clock::now();

        std::cout << global_min / (global_min + local_min) << ",";
        std::cout << local_min / (global_min + local_min) << ",";
        std::cout << std::chrono::duration_cast<std::chrono::hours> (end - begin).count() << ",";
        std::cout << std::chrono::duration_cast<std::chrono::minutes> (end - begin).count() << ",";
        std::cout << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << ",";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
    }

    return EXIT_SUCCESS;
}

//
// Created by tmac3 on 15/12/2020.
//

#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "environments/Environment.h"
#include "environments/MazeEnv.h"
#include "nodes/VarNode.h"
#include "distributions/Categorical.h"
#include "graphs/FactorGraph.h"
#include "algorithms/AlgoTree.h"
#include "algorithms/AlgoVMP.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>

using namespace hopi::environments;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::algorithms;
using namespace Eigen;

void run_simulation(MazeEnv *env, int nb_AP_steps, int nb_P_steps, bool global_inf) {
    /**
     ** Delete previous factor graph if any.
     **/
    FactorGraph::setCurrent(nullptr);

    /**
     ** Create the model's parameters.
     **/
    MatrixXd U0 = MatrixXd::Constant(env->actions(), 1, 1.0 / env->actions());
    MatrixXd A  = env->A();
    std::vector<MatrixXd> B = env->B();
    MatrixXd D0 = env->D();

    /**
     ** Create the generative model.
     **/
    VarNode *a0 = Categorical::create(U0);
    VarNode *s0 = Categorical::create(D0);
    VarNode *o0 = Transition::create(s0, A);
    o0->setType(VarNodeType::OBSERVED);
    o0->setName("o0");
    VarNode *s1 = ActiveTransition::create(s0, a0, B);
    VarNode *o1 = Transition::create(s1, A);
    o1->setName("o1");
    o1->setType(VarNodeType::OBSERVED);
    std::shared_ptr<FactorGraph> fg = FactorGraph::current();
    fg->setTreeRoot(s1);
    fg->loadEvidence(env->observations(), "../Homing-Pigeon/examples/evidences/5.evi");

    /**
     ** Create the model's prior preferences.
     **/
    MatrixXd D_tilde = MatrixXd::Constant(env->states(),  1, 1.0 / (env->states() - 1));
    MatrixXd E_tilde(env->observations(),  1);
    for (int i = 0; i < env->observations(); ++i) {
        E_tilde(i, 0) = (env->observations() - i);
    }
    E_tilde = AlgoVMP::softmax(E_tilde);

    /**
     ** Run the simulation.
     **/
    for (int i = 0; i < nb_AP_steps; ++i) { // Action perception cycle
        AlgoVMP::inference(fg->getNodes());
        auto algoTree = std::make_unique<AlgoTree>(env->actions(), D_tilde, E_tilde);
        for (int j = 0; j < nb_P_steps; ++j) { // Planning
            VarNode *n = algoTree->nodeSelection(fg);
            algoTree->expansion(n, A, B);
            if (global_inf) {
                AlgoVMP::inference(fg->getNodes());
            } else {
                AlgoVMP::inference(algoTree->lastExpansionNodes());
            }
            algoTree->evaluation();
            algoTree->backpropagation(n, fg->treeRoot());
        }
        int a = algoTree->actionSelection(fg->treeRoot());
        int o = env->execute(a);
        fg->integrate(a, fg->oneHot(env->observations(), o), A, B);
    }
}

int main()
{
    // Number of experiments
    int E = 10;
    // Number of action perception cycles
    int AP = 20;
    // Number of planning steps
    int P  = 100;
    // Number of simulations
    int N  = 100;
    // Should the inference be global?
    bool global_inf = false;

    std::cout << std::endl;
    std::cout << "Number of experiments: " << E << std::endl;
    std::cout << "Number of action perception cycles: " << AP << std::endl;
    std::cout << "Number of planning iterations: " << P << std::endl;
    std::cout << "Number of simulations: " << N << std::endl;
    std::cout << std::endl;
    std::cout << "P(solved),h,m,s,ms " << std::endl;

    for (int j = 0; j < E; ++j) { // Run the experiment E times

        double solved = 0;

        auto begin = std::chrono::steady_clock::now();

        for (int i = 0; i < N; ++i) { // For N simulations
            std::cout << i << std::endl;
            auto env = std::make_unique<MazeEnv>("../Homing-Pigeon/examples/mazes/5.maze");
            run_simulation(env.get(), AP, P, global_inf);
            auto exit_pos = env->exitPosition();
            auto agent_pos = env->agentPosition();
            if (env->manhattan_distance(agent_pos, exit_pos) <= 1) {
                solved += 1;
            }
        }

        auto end = std::chrono::steady_clock::now();

        std::cout << solved / ((double)N) << ",";
        std::cout << std::chrono::duration_cast<std::chrono::hours> (end - begin).count() << ",";
        std::cout << std::chrono::duration_cast<std::chrono::minutes> (end - begin).count() << ",";
        std::cout << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << ",";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
    }

    return EXIT_SUCCESS;
}

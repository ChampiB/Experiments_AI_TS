//
// Created by Theophile Champion on 30/07/2021.
//

#include "GraphPerformanceTracker.h"
#include "environments/Environment.h"
#include "environments/EnvType.h"
#include "environments/GraphEnv.h"

using namespace hopi::environments;

namespace experiments::trackers {

    std::unique_ptr<GraphPerformanceTracker> GraphPerformanceTracker::create() {
        return std::make_unique<GraphPerformanceTracker>();
    }

    GraphPerformanceTracker::GraphPerformanceTracker() {
        perf.push_back(0); // # of times the goal have been reached
        perf.push_back(0); // # of times the simulation was still running
        perf.push_back(0); // # of times the agent reached the bad state
    }

    void GraphPerformanceTracker::reset() {
        for (int i = 0; i < 3; ++i) {
            perf[i] = 0;
        }
    }

    void GraphPerformanceTracker::track(std::shared_ptr<Environment> &environment) {
        if (environment->type() != EnvType::GRAPH)
            throw std::runtime_error("In MazePerformanceTracker::track, invalid environment type.");
        auto env = std::dynamic_pointer_cast<GraphEnv>(environment);
        if (env->agentState() == 1)
            perf[BAD_STATE] += 1;
        else if (env->agentState() == env->goalState())
            perf[GOAL] += 1;
        else
            perf[STILL_RUNNING] += 1;
    }

    void GraphPerformanceTracker::print(std::ostream &output) const {
        double total = std::accumulate(perf.begin(), perf.end(), 0.0);
        output << "========== MAZE PERFORMANCE TRACKER ==========" << std::endl;
        output << "P(goal): " << perf[GOAL] / total << std::endl;
        output << "P(still running): " << perf[STILL_RUNNING] / total << std::endl;
        output << "P(bad state): "  << perf[BAD_STATE] / total << std::endl;
        output << std::endl;
    }

}

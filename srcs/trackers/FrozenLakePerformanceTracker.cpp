//
// Created by Theophile Champion on 01/07/2021.
//

#include <tuple>
#include <environments/EnvType.h>
#include "FrozenLakePerformanceTracker.h"
#include "environments/Environment.h"
#include "environments/FrozenLakeEnv.h"

using namespace hopi::environments;

namespace experiments::trackers {

    std::unique_ptr<FrozenLakePerformanceTracker> FrozenLakePerformanceTracker::create(int tolerance_level) {
        return std::make_unique<FrozenLakePerformanceTracker>(tolerance_level);
    }

    FrozenLakePerformanceTracker::FrozenLakePerformanceTracker(int tolerance_level) {
        tolerance = tolerance_level;
        perf = std::vector<double>(2, 0); // Reserved space for global minimum + other
        nb_fell_in_holes = 0;
    }

    void FrozenLakePerformanceTracker::reset() {
        for (double &i : perf)
            i = 0;
    }

    void FrozenLakePerformanceTracker::track(std::shared_ptr<Environment> &environment) {
        if (environment->type() != EnvType::FROZEN_LAKE)
            throw std::runtime_error("In FrozenLakePerformanceTracker::track, invalid environment type.");
        auto env = std::dynamic_pointer_cast<FrozenLakeEnv>(environment);
        auto agent_pos = env->agentPosition();
        auto exit_pos = env->exitPosition();
        double score = env->agentScore();
        double md = FrozenLakeEnv::manhattan_distance(agent_pos, exit_pos);

        score -= (md == 0) ? 10 : 0;
        nb_fell_in_holes -= (int) score;
        if (md <= tolerance)
            perf[1] += 1;
        else
            perf[0] += 1;
    }

    void FrozenLakePerformanceTracker::print(std::ostream &output) const {
        double total = std::accumulate(perf.begin(), perf.end(), 0.0);
        output << "========== FROZEN LAKE PERFORMANCE TRACKER ==========" << std::endl;
        output << "P(global): " << perf[perf.size() - 1] / total << std::endl;
        output << "P(other): " << perf[0] / total << std::endl;
        output << "Number of times the agent fell in a hole: " << nb_fell_in_holes << std::endl;
        output << std::endl;
    }

}

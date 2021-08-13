//
// Created by Theophile Champion on 01/07/2021.
//

#include <tuple>
#include <environments/EnvType.h>
#include "MazePerformanceTracker.h"
#include "environments/Environment.h"
#include "environments/MazeEnv.h"

using namespace hopi::environments;

namespace experiments::trackers {

    std::unique_ptr<MazePerformanceTracker> MazePerformanceTracker::create(
            const std::vector<std::pair<int, int>> &local_minimums_pos, int tolerance_level
    ) {
        return std::make_unique<MazePerformanceTracker>(local_minimums_pos, tolerance_level);
    }

    MazePerformanceTracker::MazePerformanceTracker(const std::vector<std::pair<int, int>> &local_minimums_pos, int tolerance_level) {
        local_pos = local_minimums_pos;
        tolerance = tolerance_level;
        perf = std::vector<double>(local_minimums_pos.size() + 2, 0); // Reserved space for # local minimums + global minimum + other
    }

    void MazePerformanceTracker::reset() {
        for (double &i : perf) {
            i = 0;
        }
    }

    void MazePerformanceTracker::track(std::shared_ptr<Environment> &environment) {
        if (environment->type() != EnvType::MAZE)
            throw std::runtime_error("In MazePerformanceTracker::track, invalid environment type.");
        auto env = std::dynamic_pointer_cast<MazeEnv>(environment);
        auto agent_pos = env->agentPosition();
        auto exit_pos = env->exitPosition();
        int local_min = -1;

        for (int i = 0; i < local_pos.size(); ++i) {
            if (MazeEnv::manhattan_distance(agent_pos, local_pos[i]) <= tolerance) {
                local_min = i;
            }
        }
        if (MazeEnv::manhattan_distance(agent_pos, exit_pos) <= tolerance) {
            perf[perf.size() - 1] += 1;
        } else if (local_min != - 1) {
            perf[local_min + 1] += 1;
        } else {
            perf[0] += 1;
        }
    }

    void MazePerformanceTracker::print(std::ostream &output) const {
        double total = std::accumulate(perf.begin(), perf.end(), 0.0);
        output << "========== MAZE PERFORMANCE TRACKER ==========" << std::endl;
        output << "P(global): " << perf[perf.size() - 1] / total << std::endl;
        for (int i = 0; i < local_pos.size(); ++i) {
            output << "P(local " + std::to_string(i + 1) + "): "  << perf[i + 1] / total << std::endl;
        }
        output << "P(other): " << perf[0] / total << std::endl;
        output << std::endl;
    }

}

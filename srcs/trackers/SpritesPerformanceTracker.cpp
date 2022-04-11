//
// Created by Theophile Champion on 01/07/2021.
//

#include <tuple>
#include <environments/EnvType.h>
#include "SpritesPerformanceTracker.h"
#include "environments/Environment.h"
#include "environments/DisentangleSpritesEnv.h"

using namespace hopi::environments;

namespace experiments::trackers {

    std::unique_ptr<SpritesPerformanceTracker> SpritesPerformanceTracker::create() {
        return std::make_unique<SpritesPerformanceTracker>();
    }

    SpritesPerformanceTracker::SpritesPerformanceTracker() {
        nb_runs = 0;
        total_perf = 0;
    }

    void SpritesPerformanceTracker::reset() {
        nb_runs = 0;
        total_perf = 0;
    }

    void SpritesPerformanceTracker::track(std::shared_ptr<Environment> &environment) {
        if (environment->type() != EnvType::D_SPRITES)
            throw std::runtime_error("In SpritesPerformanceTracker::track, invalid environment type.");
        auto env = std::dynamic_pointer_cast<DisentangleSpritesEnv>(environment);

        total_perf += env->reward_obtained();
        nb_runs += 1;
    }

    void SpritesPerformanceTracker::print(std::ostream &output) const {
        output << "========== D-SPRITES PERFORMANCE TRACKER ==========" << std::endl;
        double percentage = (total_perf + nb_runs) / (2.0 * nb_runs);
        output << "Solved percentage: " << percentage << std::endl;
        output << std::endl;
    }

}

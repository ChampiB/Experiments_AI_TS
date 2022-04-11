//
// Created by Theophile Champion on 01/07/2021.
//

#ifndef EXPERIMENTS_AI_TS_FROZEN_LAKE_PERFORMANCE_TRACKER_H
#define EXPERIMENTS_AI_TS_FROZEN_LAKE_PERFORMANCE_TRACKER_H

#include <vector>
#include <iostream>
#include "PerformanceTracker.h"

namespace experiments::trackers {

    class FrozenLakePerformanceTracker : public PerformanceTracker {
    public:
        /**
         * Create a lake performance tracker.
         * @param tolerance_level the tolerance level in distance unit
         * @return
         */
        static std::unique_ptr<FrozenLakePerformanceTracker> create(int tolerance_level = 1);

        /**
         * Constructor of the performance tracker.
         * @param tolerance_level the tolerance level in distance unit
         */
        explicit FrozenLakePerformanceTracker(int tolerance_level = 1);

        /**
         * Reset the performance tracker.
         */
        void reset() override;

        /**
         * Update the performance based on the state of the environment.
         * @param env the environment whose state determine the agent performance
         */
        void track(std::shared_ptr<hopi::environments::Environment> &env) override;

        /**
         * Display the agent performance in the output stream.
         * @param output the stream in which the performance should be written
         */
        void print(std::ostream &output) const override;

    private:
        int tolerance;
        std::vector<double> perf;
        int nb_fell_in_holes;
    };

}

#endif //EXPERIMENTS_AI_TS_FROZEN_LAKE_PERFORMANCE_TRACKER_H

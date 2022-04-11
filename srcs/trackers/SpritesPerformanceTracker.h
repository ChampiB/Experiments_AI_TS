//
// Created by Theophile Champion on 01/07/2021.
//

#ifndef EXPERIMENTS_AI_TS_SPRITES_PERFORMANCE_TRACKER_H
#define EXPERIMENTS_AI_TS_SPRITES_PERFORMANCE_TRACKER_H

#include <vector>
#include <iostream>
#include "PerformanceTracker.h"

namespace experiments::trackers {

    class SpritesPerformanceTracker : public PerformanceTracker {
    public:
        /**
         * Create a lake performance tracker.
         * @return
         */
        static std::unique_ptr<SpritesPerformanceTracker> create();

        /**
         * Constructor of the performance tracker.
         */
        explicit SpritesPerformanceTracker();

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
        int nb_runs;
        double total_perf;
    };

}

#endif //EXPERIMENTS_AI_TS_SPRITES_PERFORMANCE_TRACKER_H

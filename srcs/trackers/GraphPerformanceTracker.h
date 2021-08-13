//
// Created by Theophile Champion on 30/07/2021.
//

#ifndef EXPERIMENTS_AI_TS_GRAPH_PERFORMANCE_TRACKER_H
#define EXPERIMENTS_AI_TS_GRAPH_PERFORMANCE_TRACKER_H

#include <vector>
#include <ostream>
#include "PerformanceTracker.h"

namespace experiments::trackers {

    class GraphPerformanceTracker : public PerformanceTracker{
    public:
        enum PerfOutcome {
            GOAL = 0,
            STILL_RUNNING = 1,
            BAD_STATE = 2
        };

    public:
        /**
         * Create a graph performance tracker.
         * @return
         */
        static std::unique_ptr<GraphPerformanceTracker> create();

        /**
         * Constructor of the performance tracker.
         */
        explicit GraphPerformanceTracker();

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
        std::vector<double> perf;
    };

}

#endif //EXPERIMENTS_AI_TS_GRAPH_PERFORMANCE_TRACKER_H

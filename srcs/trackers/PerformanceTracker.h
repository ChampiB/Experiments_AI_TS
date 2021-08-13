//
// Created by Theophile Champion on 03/08/2021.
//

#ifndef EXPERIMENTS_AI_TS_PERFORMANCE_TRACKER_H
#define EXPERIMENTS_AI_TS_PERFORMANCE_TRACKER_H

#include <memory>

namespace hopi::environments {
    class Environment;
}

namespace experiments::trackers {

    class PerformanceTracker {
    public:
        /**
         * Reset the performance tracker.
         */
        virtual void reset() = 0;

        /**
         * Update the performance based on the state of the environment.
         * @param env the environment whose state determine the agent performance
         */
        virtual void track(std::shared_ptr<hopi::environments::Environment> &env) = 0;

        /**
         * Display the agent performance in the output stream.
         * @param output the stream in which the performance should be written
         */
        virtual void print(std::ostream &output) const = 0;
    };

}


#endif //EXPERIMENTS_AI_TS_PERFORMANCE_TRACKER_H

//
// Created by Theophile Champion on 01/07/2021.
//

#ifndef EXPERIMENTS_AI_TS_TIME_TRACKER_H
#define EXPERIMENTS_AI_TS_TIME_TRACKER_H

#include <iostream>
#include <memory>

namespace experiments::trackers {

    class TimeTracker {
    public:
        /**
         * Create a time tracker.
         * @return the time tracker.
         */
        static std::unique_ptr<TimeTracker> create();

        /**
         * Record the starting time point.
         */
        void tic();

        /**
         * Record the stopping time point.
         */
        void toc();

        /**
         * Display the difference between the starting and stopping time points.
         * @param output the output stream in which the display must be done.
         */
        void print(std::ostream &output) const;

    private:
        std::chrono::time_point<std::chrono::steady_clock> begin;
        std::chrono::time_point<std::chrono::steady_clock> end;
    };

}

#endif //EXPERIMENTS_AI_TS_TIME_TRACKER_H

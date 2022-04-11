//
// Created by Theophile Champion on 01/07/2021.
//

#ifndef EXPERIMENTS_AI_TS_TIME_TRACKER_H
#define EXPERIMENTS_AI_TS_TIME_TRACKER_H

#include <iostream>
#include <memory>
#include <torch/torch.h>

namespace experiments::trackers {

    class TimeTracker {
    public:
        /**
         * Create a time tracker.
         * @param nb_runs the number of runs whose time should be tracked.
         * @return the time tracker.
         */
        static std::unique_ptr<TimeTracker> create(int nb_runs);

        /**
         * Constructor.
         * @param nb_runs the number of runs whose time should be tracked.
         */
        explicit TimeTracker(int nb_runs);

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
        torch::Tensor ex_times_s{};
        torch::Tensor ex_times_ms{};
        int index;
    };

}

#endif //EXPERIMENTS_AI_TS_TIME_TRACKER_H

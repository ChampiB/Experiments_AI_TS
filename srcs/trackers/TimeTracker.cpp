//
// Created by Theophile Champion on 01/07/2021.
//

#include <chrono>
#include "TimeTracker.h"
#include "api/API.h"

using namespace hopi::api;

namespace experiments::trackers {

    std::unique_ptr<TimeTracker> TimeTracker::create(int nb_runs) {
        return std::make_unique<TimeTracker>(nb_runs);
    }

    TimeTracker::TimeTracker(int nb_runs) {
        ex_times_s = API::zeros({nb_runs});
        ex_times_ms = API::zeros({nb_runs});
        index = 0;
    }

    void TimeTracker::tic() {
        begin = std::chrono::steady_clock::now();
    }

    void TimeTracker::toc() {
        end = std::chrono::steady_clock::now();
        ex_times_s[index] = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
        ex_times_ms[index] = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        index++;
    }

    void TimeTracker::print(std::ostream &output) const {
        output << "========== TIME TRACKER ==========" << std::endl;
        output << "Time in second: " << ex_times_s.mean().item<double>() << " +/- " << ex_times_s.std().item<double>() << std::endl;
        output << "Time in millisecond: " << ex_times_ms.mean().item<double>() << " +/- " << ex_times_ms.std().item<double>() << std::endl;
        output << std::endl;
    }

}
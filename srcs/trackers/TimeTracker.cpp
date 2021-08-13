//
// Created by Theophile Champion on 01/07/2021.
//

#include <chrono>
#include "TimeTracker.h"

namespace experiments::trackers {

    std::unique_ptr<TimeTracker> TimeTracker::create() {
        return std::make_unique<TimeTracker>();
    }

    void TimeTracker::tic() {
        begin = std::chrono::steady_clock::now();
    }

    void TimeTracker::toc() {
        end = std::chrono::steady_clock::now();
    }

    void TimeTracker::print(std::ostream &output) const {
        output << "========== TIME TRACKER ==========" << std::endl;
        output << "Runtime in hours: "        << std::chrono::duration_cast<std::chrono::hours>        (end - begin).count() << std::endl;
        output << "Runtime in minutes: "      << std::chrono::duration_cast<std::chrono::minutes>      (end - begin).count() << std::endl;
        output << "Runtime in seconds: "      << std::chrono::duration_cast<std::chrono::seconds>      (end - begin).count() << std::endl;
        output << "Runtime in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
        output << std::endl;
    }

}
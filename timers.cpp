#include <chrono>

#include "timers.h"

timestamp_type timestamp() {
    return clock_type::now();
}

double duration(timestamp_type const& t) {
    return duration_type(clock_type::now()-t).count();
}

double duration(timestamp_type const& begin, timestamp_type const& end) {
    return duration_type(end-begin).count();
}

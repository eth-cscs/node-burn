#pragma once

#include <chrono>

// aliases for types used in timing host code
using clock_type    = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double>;
using timestamp_type = decltype(clock_type::now());

// return current time point
timestamp_type timestamp();

// return the time in seconds since timestamp t
double duration(timestamp_type const& t);

// return the time in seconds between two timestamps
double duration(timestamp_type const& begin, timestamp_type const& end);


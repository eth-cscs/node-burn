#include <charconv>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <cstdlib>
#include <stdio.h>

#include <fmt/core.h>

#include "instrument.h"

struct energy_sample {
    // energy in joules
    std::uint64_t energy;
    // time in us
    std::uint64_t time;
};

double average_power(const energy_sample& b, const energy_sample& e) {
    auto delta_t = (e.time - b.time)*1e-6;
    auto delta_e = e.energy - b.energy;
    return static_cast<double>(delta_e) / static_cast<double>(delta_t);
}

struct pm_probe {
    const char* file = nullptr;
    const char* name = nullptr;
};

std::vector<pm_probe> all_probes = {
    {"/sys/cray/pm_counters/energy",        "total"},
    {"/sys/cray/pm_counters/memory",        "memory"},
    {"/sys/cray/pm_counters/cpu_energy",    "cpu"},
    {"/sys/cray/pm_counters/accel0_energy", "gpu0"},
    {"/sys/cray/pm_counters/accel1_energy", "gpu1"},
    {"/sys/cray/pm_counters/accel2_energy", "gpu2"},
    {"/sys/cray/pm_counters/accel3_energy", "gpu3"},
    {"/sys/cray/pm_counters/cpu0_energy",   "cpu0"},
    {"/sys/cray/pm_counters/cpu1_energy",   "cpu1"},
    {"/sys/cray/pm_counters/cpu2_energy",   "cpu2"},
    {"/sys/cray/pm_counters/cpu3_energy",   "cpu3"},
};

std::optional<energy_sample> read_energy(const char* fname) {
    energy_sample e;

    FILE* file = fopen(fname, "r");

    if (file == nullptr) {
        return std::nullopt;
    }
    if (fscanf(file, "%llu %*s %llu %*s", &e.energy, &e.time) != 2) {
        fclose(file);
        return std::nullopt;
    }
    return e;
}

std::optional<int> get_local_slurm_id() {
    const char* value = std::getenv("SLURM_LOCALID");
    if (value) {
        int id;
        auto [ptr, ec] = std::from_chars(value, value + std::strlen(value), id);
        if (ec == std::errc()) {
            return id;
        }
    }
    return std::nullopt;
}

struct pm_counters: public instrument {
    bool enabled_ = false;
    const char* err_string_ = nullptr;
    std::vector<pm_probe> probes_;
    std::vector<energy_sample> start_;
    std::vector<energy_sample> end_;

    pm_counters() {
        // check that pm_counters are available
        probes_.reserve(all_probes.size());
        start_.reserve(all_probes.size());
        end_.reserve(all_probes.size());
        for (const auto& probe: all_probes) {
            if (std::filesystem::exists(probe.file)) {
                probes_.push_back(probe);
            }
        }
        enabled_ = !probes_.empty();
        if (!enabled_) {
            err_string_ = "no energy counters available in /sys/cray/pm_counters";
        }
    }

    // return true if successful
    bool is_enabled() override {
        return enabled_;
    }

    // start instrumentation
    virtual void start() override {
        for (const auto& probe: probes_) {
            auto sample = read_energy(probe.file);
            if (sample) {
                start_.push_back(*sample);
            }
            else {
                enabled_=false;
                break;
            }
        }
    }

    // stop instrumentation
    virtual void stop() override {
        if (enabled_) {
            for (const auto& probe: probes_) {
                auto sample = read_energy(probe.file);
                if (sample) {
                    end_.push_back(*sample);
                }
                else {
                    enabled_=false;
                    break;
                }
            }
        }
    }

    // generate one line string that describes the result
    void print_result(const std::string& prefix) override {
        if (!enabled_) {
            fmt::print("{}{}", prefix, err_string_);
            return;
        }
        std::string values;
        for (unsigned i=0; i<probes_.size(); ++i) {
            const auto& probe = probes_[i];
            values += fmt::format("{}{} {:.0f}", (i==0?"":", "), probe.name, average_power(start_[i], end_[i]));
        }
        // This method prints the energy counters for the whole node, ideally it is only
        // printed once per node.
        // Print the results if either if this is not a slurm job, or if this is rank has
        // local rank of 0 on the node.
        if (auto local_id = get_local_slurm_id(); !local_id || *local_id==0) {
            fmt::print("{}power [{}]\n", prefix, values);
        }
    }
};

std::unique_ptr<instrument> get_instrument() {
    return std::make_unique<pm_counters>();
}

/*

   The following energy probes are available on the different HPE Cray-EX nodes

# gh200

energy
accel0_energy
accel1_energy
accel2_energy
accel3_energy
cpu0_energy
cpu1_energy
cpu2_energy
cpu3_energy
cpu_energy

# a100

energy
accel0_energy
accel1_energy
accel2_energy
accel3_energy
cpu_energy
memory_energy


# zen2

energy
cpu_energy
memory_energy

# mi200

energy
accel0_energy
accel1_energy
accel2_energy
accel3_energy
cpu_energy
memory_energy

*/

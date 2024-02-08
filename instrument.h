#pragma once

#include <memory>
#include <string>

struct instrument {
    // return true if successful
    virtual bool is_enabled() {return false;};

    // start instrumentation
    virtual void start() {};

    // stop instrumentation
    virtual void stop() {};

    // generate one line string that describes the result
    virtual void print_result(const std::string& prefix) {};

    virtual ~instrument() {};
};


#ifdef NB_INSTRUMENT
std::unique_ptr<instrument> get_instrument();
#else
std::unique_ptr<instrument> get_instrument() {
    return std::make_unique<instrument>();
}
#endif

#ifndef PICA_DEMO_UTILITY_TIMER_H
#define PICA_DEMO_UTILITY_TIMER_H


namespace utility {

// An abstract stopwatch, for high-precision timing.
struct Stopwatch
{
    // Reset the accumulated time to 0 seconds.
    virtual void reset() = 0;
    // Start timing.
    virtual void start() = 0;
    // Stop timing, add the time elapsed since the last start to the
    // accumulated time.
    virtual void stop() = 0;
    // Return the accumulated time, in seconds.
    virtual double getElapsed() = 0;
};

// Create a stopwatch suitable for this system. The stopwatch is initially
// reset to 0.
Stopwatch * createStopwatch();

} // namespace utility


#endif

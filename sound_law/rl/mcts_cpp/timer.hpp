#pragma once

#include "common.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;
typedef std::chrono::_V2::system_clock::time_point time_point;

class Timer;

class TimerScope
{
    friend class Timer;

    Timer &timer;
    time_point start_time;
    std::string name;

    TimerScope(Timer &, const std::string &);

public:
    ~TimerScope();
};

class Timer
{
    friend class TimerScope;

    ParaMap<std::string, float> elapsed;
    bool enabled = false;
    bool started = false;

    inline void update(const std::string &, float);

public:
    TimerScope start(const std::string &);
    void enable();
    void disable();
    void show_stats();
};

extern Timer timer;

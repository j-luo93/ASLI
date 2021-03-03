#pragma once

#include "common.hpp"
#include "timer.hpp"

class Stats
{
    int value;
    int weight;
    Timer timer;

public:
    Stats();

    void update(int);
    void show_stats();
    void enable_timer();
    void disable_timer();
    TimerScope start_timer(const std::string &);
};

extern Stats stats;
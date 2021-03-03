#include "stats.hpp"

Stats::Stats() { timer = Timer(); }

void Stats::update(int inc)
{
    value += inc;
    weight++;
}

void Stats::show_stats()
{
    std::cerr << "#stats: \n";
    std::cerr << value << " / " << weight << " = " << static_cast<float>(value) / static_cast<float>(weight) << '\n';
    timer.show_stats();
}

void Stats::enable_timer() { timer.enable(); }
void Stats::disable_timer() { timer.disable(); }
TimerScope Stats::start_timer(const std::string &name) { return timer.start(name); }

Stats stats = Stats();
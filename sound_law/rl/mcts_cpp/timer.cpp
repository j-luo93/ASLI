#include "timer.hpp"

TimerScope::TimerScope(Timer &timer,
                       const std::string &name) : timer(timer),
                                                  start_time(Time::now()),
                                                  name(name) {}

TimerScope::~TimerScope()
{
    if (timer.enabled)
    {
        fsec fs = Time::now() - start_time;
        timer.update(name, fs.count());
    }
}

void Timer::update(const std::string &name, float count)
{
    elapsed.try_emplace_l(
        name, [count](float &value) { value += count; }, count);
}

TimerScope Timer::start(const std::string &name) { return TimerScope(*this, name); }

void Timer::enable()
{
    enabled = true;
    std::cerr << "timer enabled.\n";
}

void Timer::disable()
{
    enabled = false;
    std::cerr << "timer disabled.\n";
}

void Timer::show_stats()
{
    float total = 0.0;
    for (const auto &item : elapsed)
        total += item.second;
    std::cerr << elapsed.size() << " stats available. Total time elapsed " << total << '\n';
    for (const auto &item : elapsed)
    {
        std::cerr << item.first << ": " << item.second << "s (" << item.second / total * 100 << "%) elapsed.\n";
    }
}

Timer timer = Timer();
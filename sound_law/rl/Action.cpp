#include <Action.h>

Action::Action(long action_id, long before_id, long after_id)
{
    this->action_id = action_id;
    this->before_id = before_id;
    this->after_id = after_id;
}
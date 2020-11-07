#include <Action.h>

Action::Action(uint action_id, uint before_id, uint after_id)
{
    this->action_id = action_id;
    this->before_id = before_id;
    this->after_id = after_id;
}
#include <ActionSpace.h>

ActionSpace::ActionSpace()
{
    this->actions = vector<Action *>();
    this->word_cache = unordered_map<string, Word *>();
    this->uni_map = unordered_map<long, vector<long>>();
}

void ActionSpace::register_action(long before_id, long after_id)
{
    long action_id = this->actions.size();
    Action *action = new Action(action_id, before_id, after_id);
    this->actions.push_back(action);
    if (this->uni_map.find(before_id) == this->uni_map.end())
    {
        this->uni_map[before_id] = vector<long>();
    }
    this->uni_map[before_id].push_back(action_id);
}

Action *ActionSpace::get_action(long action_id)
{
    return this->actions[action_id];
}

vector<long> ActionSpace::get_action_allowed(VocabIdSeq vocab_i)
{
    unordered_set<long> action_set = unordered_set<long>();
    for (long i = 0; i < vocab_i.size(); ++i)
    {
        string key = get_key(vocab_i[i]);
        unique_lock<mutex> lock(this->mtx);
        if (this->word_cache.find(key) == this->word_cache.end())
        {
            Word *new_word = new Word(vocab_i[i]);
            this->word_cache[new_word->key] = new_word;
            unordered_set<long> new_aa = unordered_set<long>();
            for (auto const &item : this->uni_map)
            {
                long uni_key = item.first;
                if (new_word->uni_keys.find(uni_key) != new_word->uni_keys.end())
                {
                    vector<long> map_values = item.second;
                    new_aa.insert(map_values.begin(), map_values.end());
                }
            }
            new_word->action_allowed = new_aa;
        }
        Word *word = this->word_cache[key];
        lock.unlock();
        unordered_set<long> aa = word->action_allowed;
        action_set.insert(aa.begin(), aa.end());
    }
    return vector<long>(action_set.begin(), action_set.end());
}

long ActionSpace::size()
{
    return this->actions.size();
}
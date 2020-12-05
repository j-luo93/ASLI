#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <assert.h>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

using abc_t = int16_t;     // for alphabets -- this could be negative since -1 is used for marking null.
using action_t = uint32_t; // for actions
using Action = std::array<abc_t, 6>;
using IdSeq = std::vector<abc_t>;
using VocabIdSeq = std::vector<IdSeq>;

static const abc_t NULL_abc = -1;

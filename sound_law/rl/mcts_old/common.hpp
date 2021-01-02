#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <array>
#include <limits>
#include <assert.h>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread.hpp>

#include "robin_hood.h"

// using abc_t = int16_t;     // for alphabets -- this could be negative since -1 is used for marking null.
// using action_t = uint32_t; // for actions
// using visit_t = int32_t;   // for visit/action counts -- due to virtual games, this could be negative.
// using tn_cnt_t = uint64_t; // for tree nodes
// using Action = std::array<abc_t, 6>;
// using IdSeq = std::vector<abc_t>;
// using VocabIdSeq = std::vector<IdSeq>;
// using Site = std::array<abc_t, 5>;

// static const abc_t NULL_abc = -1;
// static const action_t NULL_action = std::numeric_limits<action_t>::max();

// template <class T, class V>
// using UMap = boost::unordered_map<T, V>;

// template <class V>
// using ActionMap = UMap<Action, V>;

using abc_t = uint16_t;
using visit_t = int32_t;   // for visit/action counts -- due to virtual games, this could be negative.
using tn_cnt_t = uint64_t; // for tree nodes
using IdSeq = std::vector<abc_t>;
using VocabIdSeq = std::vector<IdSeq>;

using uai_t = uint64_t; // for actions -- short for unique action identifier
using usi_t = uint64_t; // for sites -- short for unique site identifier

// template <class K, class V>
// using UMap = boost::unordered_map<K, V>;

// template <class K>
// using USet = boost::unordered_set<K>;

template <class K, class V>
using UMap = robin_hood::unordered_map<K, V>;

template <class K>
using USet = robin_hood::unordered_set<K>;

template <class V>
using ActionMap = UMap<uai_t, V>;

static const uint64_t last_10 = (1 << 10) - 1;
static const abc_t NULL_ABC = static_cast<abc_t>(last_10);
static const uai_t NULL_ACTION = std::numeric_limits<uai_t>::max();

namespace action
{
    // Each action has the following format:
    // <pre_id> <d_pre_id> <post_id> <d_post_id> <before_id> <after_id>
    // Each site has a similar format but without the last <after_id>.

    // Obtain a UAI by combining a USI with after_id;
    inline uai_t combine_after_id(usi_t site, abc_t after_id) { return ((site << 10) | static_cast<abc_t>(after_id)); }
    inline uai_t combine(abc_t pre_id, abc_t d_pre_id, abc_t post_id, abc_t d_post_id, abc_t before_id, abc_t after_id)
    {
        return ((static_cast<uai_t>(pre_id) << 50) |
                (static_cast<uai_t>(d_pre_id) << 40) |
                (static_cast<uai_t>(post_id) << 30) |
                (static_cast<uai_t>(d_post_id) << 20) |
                (static_cast<uai_t>(before_id) << 10) |
                static_cast<uai_t>(after_id));
    }

    inline abc_t get_after_id(uai_t action) { return static_cast<abc_t>(action & last_10); }
    inline abc_t get_before_id(uai_t action) { return static_cast<abc_t>((action >> 10) & last_10); }
    inline abc_t get_d_post_id(uai_t action) { return static_cast<abc_t>((action >> 20) & last_10); }
    inline abc_t get_post_id(uai_t action) { return static_cast<abc_t>((action >> 30) & last_10); }
    inline abc_t get_d_pre_id(uai_t action) { return static_cast<abc_t>((action >> 40) & last_10); }
    inline abc_t get_pre_id(uai_t action) { return static_cast<abc_t>(action >> 50); }
    static const uai_t STOP = combine(NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC);

} // namespace action

namespace site
{
    inline abc_t get_before_id(usi_t site) { return static_cast<abc_t>(site & last_10); }
    inline usi_t combine(abc_t pre_id, abc_t d_pre_id, abc_t post_id, abc_t d_post_id, abc_t before_id)
    {
        return ((static_cast<usi_t>(pre_id) << 40) |
                (static_cast<usi_t>(d_pre_id) << 30) |
                (static_cast<usi_t>(post_id) << 20) |
                (static_cast<usi_t>(d_post_id) << 10) |
                static_cast<usi_t>(before_id));
    }
} // namespace site
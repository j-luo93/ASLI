#pragma once

#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <limits>
#include <atomic>
#include <assert.h>
#include <boost/functional/hash.hpp>
#include "parallel-hashmap/parallel_hashmap/phmap.h"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

#include "ctpl.h"

using abc_t = uint16_t;
using visit_t = int32_t; // for visit/action counts -- due to virtual games, this could be negative.
// using tn_cnt_t = uint64_t; // for tree nodes
using IdSeq = std::vector<abc_t>;
using VocabIdSeq = std::vector<IdSeq>;

namespace std
{
    template <>
    class hash<IdSeq>
    {
        boost::hash<IdSeq> hasher = boost::hash<IdSeq>();

    public:
        inline size_t operator()(const IdSeq &k) const
        {
            return hasher(k);
        }
    };
}; // namespace std

using uai_t = uint64_t; // for actions -- short for unique action identifier
using usi_t = uint64_t; // for sites -- short for unique site identifier

template <class K, class V>
using UMap = phmap::flat_hash_map<K, V>;

template <class K>
using USet = phmap::flat_hash_set<K>;

template <class T, class V>
using ParaMap = phmap::parallel_flat_hash_map<T, V,
                                              std::hash<T>,
                                              std::equal_to<T>,
                                              std::allocator<std::pair<const T, V>>,
                                              4,
                                              std::mutex>;

template <class V>
using ActionMap = UMap<uai_t, V>;

template <class V>
using vec = std::vector<V>;

template <class... TupleArgs>
using tup = std::tuple<TupleArgs...>;

template <class T1, class T2>
using pair = std::pair<T1, T2>;

using Pool = ctpl::thread_pool;

template <class T>
inline void show_size(const T &obj, std::string msg) { SPDLOG_INFO("{0} size: {1}", msg, obj.size()); }

constexpr uint64_t last_10 = (static_cast<uint64_t>(1) << 10) - 1;
constexpr uint64_t first_4 = ((static_cast<uint64_t>(1) << 4) - 1) << 60;
constexpr abc_t NULL_ABC = static_cast<abc_t>(last_10);
constexpr uai_t NULL_ACTION = std::numeric_limits<uai_t>::max();
constexpr usi_t NULL_SITE = std::numeric_limits<usi_t>::max();

template <class... TupleArgs>
auto get_tuple_at(size_t index, const std::tuple<TupleArgs...> &inputs_tuple)
{
    return std::apply([index](auto &&... item) { return std::forward_as_tuple(item[index]...); }, inputs_tuple);
}

template <bool in_place, class F, class O, class I0, class... Is>
void parallel_apply_sequential(const F &func, std::vector<O> &outputs, const std::vector<I0> &inputs, const Is &... rest)
{
    size_t n = inputs.size();
    outputs.resize(n);
    if constexpr (in_place)
    {
        auto inputs_tuple = std::forward_as_tuple(outputs, inputs, rest...);
        for (size_t i = 0; i < n; i++)
            std::apply(func, get_tuple_at(i, inputs_tuple));
    }
    else
    {
        auto inputs_tuple = std::forward_as_tuple(inputs, rest...);
        for (size_t i = 0; i < n; i++)
            outputs[i] = std::apply(func, get_tuple_at(i, inputs_tuple));
    }
}

template <bool in_place, size_t CS = 0, class F, class O, class I0, class... Is>
void parallel_apply(Pool *tp, const F &func, std::vector<O> &outputs, const std::vector<I0> &inputs, const Is &... rest)
{
    if (tp == nullptr)
    {
        parallel_apply_sequential<in_place>(func, outputs, inputs, rest...);
        return;
    }

    size_t n = inputs.size();
    outputs.resize(n);

    size_t num_chunks, chunk_size;
    if constexpr (CS == 0)
    {
        num_chunks = tp->size();
        chunk_size = n / num_chunks + ((n % num_chunks > 0) ? 1 : 0);
    }
    else
    {
        chunk_size = CS;
        num_chunks = n / chunk_size + ((n % chunk_size > 0) ? 1 : 0);
    }
    vec<std::future<void>> results(num_chunks);
    if constexpr (in_place)
    {
        auto inputs_tuple = std::forward_as_tuple(outputs, inputs, rest...);
        for (size_t i = 0; i < num_chunks; i++)
        {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            results[i] = tp->push(
                [start, end, &func, &inputs_tuple](int) {
                    for (size_t j = start; j < end; j++)
                        std::apply(func, get_tuple_at(j, inputs_tuple));
                });
        }
    }
    else
    {
        auto inputs_tuple = std::forward_as_tuple(inputs, rest...);
        for (size_t i = 0; i < num_chunks; i++)
        {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            results[i] = tp->push(
                [start, end, &func, &outputs, &inputs_tuple](int) {
                    for (size_t j = start; j < end; j++)
                        outputs[j] = std::apply(func, get_tuple_at(j, inputs_tuple));
                });
        }
    }
    for (size_t i = 0; i < num_chunks; i++)
        results[i].wait();
}

template <size_t CS = 0, class F, class I0, class... Is>
void parallel_apply(Pool *tp, const F &func, const std::vector<I0> &inputs, const Is &... rest)
{
    auto dummy = vec<bool>();
    parallel_apply<false, CS>(
        tp, [&func](auto... args) -> bool { std::apply(func, std::make_tuple(args...)); return true; }, dummy, inputs, rest...);
}

template <class K, class F>
void find_unique(vec<K> &outputs, const vec<K> &inputs, const F &filter)
{
    outputs.reserve(inputs.size());
    auto added = USet<K>();
    for (auto input : inputs)
    {
        if ((added.find(input) == added.end()) && filter(input))
        {
            added.insert(input);
            outputs.push_back(input);
        }
    }
}

// FIXME(j_luo) merge them.
template <class K, class V>
void find_unique(vec<K> &outputs, const vec<K> &inputs, const UMap<K, V> &cache)
{
    find_unique(outputs, inputs, [&cache](K key) { return cache.find(key) == cache.end(); });
}

template <class K, class V>
void find_unique(vec<K> &outputs, const vec<vec<K>> &inputs, const UMap<K, V> &cache)
{
    size_t total_size = 0;
    for (const auto &item : inputs)
        total_size += item.size();
    outputs.reserve(total_size);
    auto added = USet<K>();
    for (const auto &inner : inputs)
        for (auto input : inner)
        {
            if ((added.find(input) == added.end()) && (cache.find(input) == cache.end()))
            {
                added.insert(input);
                outputs.push_back(input);
            }
        }
}

enum class SpecialType : uai_t
{
    NONE = static_cast<uai_t>(0),
    CLL = (static_cast<uai_t>(1) << 60),
    CLR = (static_cast<uai_t>(2) << 60),
    VS = (static_cast<uai_t>(3) << 60),
    GBJ = (static_cast<uai_t>(4) << 60),
    GBW = (static_cast<uai_t>(5) << 60)
};

enum class Stress : int
{
    NOSTRESS,
    STRESSED,
    UNSTRESSED
};

namespace action
{
    // Each action has the following format:
    // <pre_id> <d_pre_id> <post_id> <d_post_id> <before_id> <after_id>
    // Each site has a similar format but without the last <after_id>.
    inline abc_t get_after_id(uai_t action) { return static_cast<abc_t>(action & last_10); }
    inline abc_t get_before_id(uai_t action) { return static_cast<abc_t>((action >> 10) & last_10); }
    inline abc_t get_d_post_id(uai_t action) { return static_cast<abc_t>((action >> 20) & last_10); }
    inline abc_t get_post_id(uai_t action) { return static_cast<abc_t>((action >> 30) & last_10); }
    inline abc_t get_d_pre_id(uai_t action) { return static_cast<abc_t>((action >> 40) & last_10); }
    inline abc_t get_pre_id(uai_t action) { return static_cast<abc_t>((action >> 50) & last_10); }
    inline SpecialType get_special_type(uai_t action) { return static_cast<SpecialType>(action & first_4); }

    // Obtain a UAI by combining a USI with after_id;
    inline usi_t get_site(uai_t action) { return action >> 10; }
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
    static const uai_t STOP = combine(NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC);
    inline uai_t combine_special(abc_t pre_id, abc_t d_pre_id, abc_t post_id, abc_t d_post_id, abc_t before_id, abc_t after_id, SpecialType st)
    {
        return (combine(pre_id, d_pre_id, post_id, d_post_id, before_id, after_id) | static_cast<uai_t>(st));
    }

    inline std::string str(uai_t action)
    {
        std::string pre = "( " + std::to_string(get_d_pre_id(action)) + " + " + std::to_string(get_pre_id(action)) + " )";
        std::string post = "( " + std::to_string(get_post_id(action)) + " + " + std::to_string(get_d_post_id(action)) + " )";
        return pre + " + " + std::to_string(get_before_id(action)) + " + " + post + " > " + std::to_string(get_after_id(action));
    }
} // namespace action

namespace site
{
    inline abc_t get_before_id(usi_t site) { return static_cast<abc_t>(site & last_10); }
    inline abc_t get_d_post_id(usi_t site) { return static_cast<abc_t>((site >> 10) & last_10); }
    inline abc_t get_post_id(usi_t site) { return static_cast<abc_t>((site >> 20) & last_10); }
    inline abc_t get_d_pre_id(usi_t site) { return static_cast<abc_t>((site >> 30) & last_10); }
    inline abc_t get_pre_id(usi_t site) { return static_cast<abc_t>(site >> 40); }
    inline usi_t combine(abc_t pre_id, abc_t d_pre_id, abc_t post_id, abc_t d_post_id, abc_t before_id)
    {
        return ((static_cast<usi_t>(pre_id) << 40) |
                (static_cast<usi_t>(d_pre_id) << 30) |
                (static_cast<usi_t>(post_id) << 20) |
                (static_cast<usi_t>(d_post_id) << 10) |
                static_cast<usi_t>(before_id));
    }

    inline std::string str(usi_t site)
    {
        std::string pre = "( " + std::to_string(get_d_pre_id(site)) + " + " + std::to_string(get_pre_id(site)) + " )";
        std::string post = "( " + std::to_string(get_post_id(site)) + " + " + std::to_string(get_d_post_id(site)) + " )";
        return pre + " + " + std::to_string(get_before_id(site)) + " + " + post;
    }
} // namespace site

class DistTable
{
    const size_t size;
    vec<std::atomic<float>> data;

public:
    inline DistTable(size_t size) : size(size), data(vec<std::atomic<float>>(size))
    {
        for (size_t i = 0; i < size; i++)
            set(i, -1.0);
    }
    inline float get(size_t index)
    {
        assert(index < size);
        assert(data[index] >= 0);
        return data[index];
    }
    inline std::atomic<float> *locate(size_t index)
    {
        assert(index < size);
        return &data[index];
    }
    inline void set(size_t index, float value) { data[index] = value; }
};

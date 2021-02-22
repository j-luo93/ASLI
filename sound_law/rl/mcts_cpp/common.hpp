#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <algorithm>
#include <boost/functional/hash.hpp>

#include "ctpl.h"
#include "parallel-hashmap/parallel_hashmap/phmap.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

using abc_t = uint16_t;
using visit_t = int32_t; // for visit/action counts -- due to virtual games, this could be negative.`

using string = std::string;

template <class T>
using vec = std::vector<T>;

using Pool = ctpl::thread_pool;

template <class K, class V>
using map = phmap::flat_hash_map<K, V>;

template <class T, class V>
using paramap = phmap::parallel_flat_hash_map<T, V,
                                              std::hash<T>,
                                              std::equal_to<T>,
                                              std::allocator<std::pair<const T, V>>,
                                              4,
                                              std::mutex>;

template <class K>
using set = phmap::flat_hash_set<K>;

template <class T1, class T2>
using pair = std::pair<T1, T2>;

template <class... TupleArgs>
using tuple = std::tuple<TupleArgs...>;

template <class T, size_t S>
using array = std::array<T, S>;

template <class T>
using list = std::list<T>;

using IdSeq = vec<abc_t>;
using VocabIdSeq = vec<IdSeq>;

constexpr uint64_t last_10 = (static_cast<uint64_t>(1) << 10) - 1;

// namespace abc
// {
//     constexpr abc_t NONE = 0;
// };

// Define a hasher for IdSeq in std namespace by borrowing from boost.
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

template <class K, class F>
vec<K> find_unique(const vec<K> &inputs, const F &filter)
{
    auto outputs = vec<K>();
    outputs.reserve(inputs.size());
    auto added = set<K>();
    for (const auto &input : inputs)
    {
        if ((!added.contains(input)) && filter(input))
        {
            added.insert(input);
            outputs.push_back(input);
        }
    }
    return outputs;
}

template <class T>
inline void show_size(const T &obj, std::string msg) { SPDLOG_INFO("{0} size: {1}", msg, obj.size()); }

namespace str
{
    inline string from(bool b) { return b ? "true" : "false"; }
} // namespace str

enum class Stress : int
{
    NOSTRESS,
    STRESSED,
    UNSTRESSED
};

enum class SpecialType : abc_t // Treat it as a special char to compatibility with ChosenChar in MiniNode.
{
    NONE,
    CLL,
    CLR,
    VS,
    GBJ,
    GBW
};

template <class K, class V>
class Trie;

template <class K, class V>
class TrieNode
{
    friend class Trie<K, V>;
    paramap<K, TrieNode<K, V> *> children;
    V value;
};

template <class K, class V>
class Trie
{
    TrieNode<K, V> *root;
    // Get the value associated with a key. If the key already exists, return true and modify the argument `value` by reference.
    // If it doesn't exist, return false and insert the argument `value` into the trie.
public:
    Trie() : root(new TrieNode<K, V>()){};

    bool get(const vec<K> &key, V &new_value)
    {
        TrieNode<K, V> *node = root;
        bool inserted = false;
        for (const K k : key)
            if (!node->children.if_contains(k, [&node](TrieNode<K, V> *const &value) { node = value; }))
            {
                auto new_node = new TrieNode<K, V>();
                node->children.try_emplace_l(
                    k, [&new_node](TrieNode<K, V> *&value) {delete new_node; new_node = value; }, new_node);
                node = new_node;
                inserted = true;
            }
        if (inserted)
        {
            node->value = new_value;
            return false;
        }
        else
        {
            new_value = node->value;
            return true;
        }
    };
};

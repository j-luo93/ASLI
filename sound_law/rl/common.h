#pragma once

#include <vector>
#include <list>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <assert.h>
#include <iostream>
#include <limits>

using namespace std;

// Define basic integer types for different purporses.
using abc_t = uint16_t;    // for alphabet
using cost_t = float;      // for cost
using dist_t = float;      // for distance
using visit_t = int32_t;   // for visit/action counts -- due to virtual games, this could be negative.
using action_t = uint32_t; // for actions
using node_t = uint64_t;   // for node id

// Use the maximum values as the sentinel/null values.
abc_t NULL_abc = numeric_limits<abc_t>::max();
action_t NULL_action = numeric_limits<action_t>::max();

using IdSeq = vector<abc_t>;
using VocabIdSeq = vector<IdSeq>;

dist_t edit_distance(const IdSeq &seq1, const IdSeq &seq2, const vector<vector<cost_t>> &dist_mat, cost_t ins_cost)
{
    size_t l1 = seq1.size();
    size_t l2 = seq2.size();
    dist_t **dist = (dist_t **)malloc((l1 + 1) * sizeof(dist_t **));
    for (size_t i = 0; i < l1 + 1; ++i)
        dist[i] = (dist_t *)malloc((l2 + 1) * sizeof(dist_t *));

    for (size_t i = 0; i < l1 + 1; ++i)
        dist[i][0] = i * ins_cost;
    for (size_t i = 0; i < l2 + 1; ++i)
        dist[0][i] = i * ins_cost;

    cost_t sub_cost;
    bool use_phono_edit_dist = (dist_mat.size() > 0);
    for (size_t i = 1; i < l1 + 1; ++i)
        for (size_t j = 1; j < l2 + 1; ++j)
        {
            if (use_phono_edit_dist)
            {
                sub_cost = dist_mat[seq1[i - 1]][seq2[j - 1]];
            }
            else
            {
                sub_cost = seq1[i - 1] == seq2[j - 1] ? 0 : 1;
            }
            dist[i][j] = min(dist[i - 1][j - 1] + sub_cost, min(dist[i - 1][j], dist[i][j - 1]) + ins_cost);
        }
    dist_t ret = dist[l1][l2];
    for (size_t i = 0; i < l1 + 1; ++i)
        free(dist[i]);
    free(dist);
    return ret;
};

using WordKey = string;

// Copied from https://gist.github.com/angeleno/e838a35f0849ecab56e8be7e46645177.
namespace std
{
    template <typename T>
    inline void hash_combine(size_t &seed, const T &val)
    {
        hash<T> hasher;
        seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template <class... TupleArgs>
    struct hash<tuple<TupleArgs...>>
    {
    private:
        //  this is a termination condition
        //  N == sizeof...(TupleTypes)
        //
        template <size_t Idx, typename... TupleTypes>
        inline typename enable_if<Idx == sizeof...(TupleTypes), void>::type
        hash_combine_tup(size_t &seed, const tuple<TupleTypes...> &tup) const
        {
        }

        //  this is the computation function
        //  continues till condition N < sizeof...(TupleTypes) holds
        //
        template <size_t Idx, typename... TupleTypes>
            inline typename enable_if < Idx<sizeof...(TupleTypes), void>::type
                                        hash_combine_tup(size_t &seed, const tuple<TupleTypes...> &tup) const
        {
            hash_combine(seed, get<Idx>(tup));

            //  on to next element
            hash_combine_tup<Idx + 1>(seed, tup);
        }

    public:
        size_t operator()(const tuple<TupleArgs...> &tupleValue) const
        {
            size_t seed = 0;
            //  begin with the first iteration
            hash_combine_tup<0>(seed, tupleValue);
            return seed;
        }
    };
} // namespace std

WordKey get_word_key(const IdSeq &id_seq)
{
    WordKey key = "";
    size_t i = 0;
    while (i < id_seq.size() - 1)
    {
        key += to_string(id_seq[i]) + ',';
        i++;
    }
    key += to_string(id_seq[i]);
    return key;
}

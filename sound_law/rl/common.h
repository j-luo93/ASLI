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
using cost_t = uint8_t;    // for cost
using dist_t = uint16_t;   // for distance
using visit_t = int16_t;   // for visit/action counts -- due to virtual games, this could be negative.
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

string get_key(const IdSeq &id_seq)
{
    string key = "";
    size_t i = 0;
    while (i < id_seq.size() - 1)
    {
        key += to_string(id_seq[i]) + ',';
        i++;
    }
    key += to_string(id_seq[i]);
    return key;
}

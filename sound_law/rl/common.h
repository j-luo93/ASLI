#pragma once

#include <vector>
#include <list>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <assert.h>

using namespace std;

// FIXME(j_luo) Probably need list for insertion speed.
// FIXME(j_luo) Use int to reduce memory?
using IdSeq = vector<long>;
using VocabIdSeq = vector<IdSeq>;

long edit_distance(const IdSeq &seq1, const IdSeq &seq2, const vector<vector<long>> &dist_mat, long ins_cost)
{
    long l1 = seq1.size();
    long l2 = seq2.size();
    long **dist = (long **)malloc((l1 + 1) * sizeof(long **));
    for (long i = 0; i < l1 + 1; ++i)
        dist[i] = (long *)malloc((l2 + 1) * sizeof(long *));

    for (long i = 0; i < l1 + 1; ++i)
        dist[i][0] = i;
    for (long i = 0; i < l2 + 1; ++i)
        dist[0][i] = i;

    long sub_cost;
    bool use_phono_edit_dist = (dist_mat.size() > 0);
    for (long i = 1; i < l1 + 1; ++i)
        for (long j = 1; j < l2 + 1; ++j)
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
    long ret = dist[l1][l2];
    for (long i = 0; i < l1 + 1; ++i)
        free(dist[i]);
    free(dist);
    return ret;
};

string get_key(const IdSeq &id_seq)
{
    string key = "";
    long i = 0;
    while (i < id_seq.size() - 1)
    {
        key += to_string(id_seq[i]) + ',';
        i++;
    }
    key += to_string(id_seq[i]);
    return key;
}

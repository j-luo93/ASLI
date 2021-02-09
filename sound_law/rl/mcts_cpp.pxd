# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

cdef extern from "mcts_cpp/word.cpp": pass
cdef extern from "mcts_cpp/action.cpp": pass
cdef extern from "mcts_cpp/env.cpp": pass
cdef extern from "mcts_cpp/node.cpp": pass
cdef extern from "mcts_cpp/mcts.cpp": pass

cdef extern from "mcts_cpp/ctpl.h": pass

cdef extern from "mcts_cpp/parallel-hashmap/parallel_hashmap/phmap.h"  namespace "phmap" nogil:
    cdef cppclass flat_hash_map[T, U]:
        pass

    cdef cppclass flat_hash_set[T]:
        pass

cdef extern from "mcts_cpp/common.hpp":
    ctypedef unsigned short abc_t
    ctypedef int visit_t
    ctypedef vector[abc_t] IdSeq
    ctypedef vector[IdSeq] VocabIdSeq

    cdef cppclass Stress:
        pass

    cdef cppclass SpecialType:
        pass

cdef extern from "mcts_cpp/common.hpp" namespace "Stress":
    cdef Stress NOSTRESS
    cdef Stress STRESSED
    cdef Stress UNSTRESSED

cdef extern from "mcts_cpp/common.hpp" namespace "SpecialType":
    cdef SpecialType NONE
    cdef SpecialType CLL
    cdef SpecialType CLR
    cdef SpecialType VS
    cdef SpecialType GBJ
    cdef SpecialType GBW

cdef extern from "mcts_cpp/word.hpp":
    cdef cppclass Word nogil:
        IdSeq id_seq

    cdef cppclass WordSpaceOpt nogil:
        vector[vector[float]] dist_mat
        float ins_cost
        vector[bool] is_vowel
        vector[Stress] unit_stress
        vector[abc_t] unit2base
        vector[abc_t] unit2stressed
        vector[abc_t] unit2unstressed

        WordSpaceOpt()

    cdef cppclass WordSpace nogil:
        WordSpaceOpt opt

ctypedef Word * Wptr

cdef extern from "mcts_cpp/action.hpp":
    cdef cppclass ActionSpaceOpt nogil:
        abc_t null_id
        abc_t emp_id
        abc_t sot_id
        abc_t eot_id
        abc_t any_id
        abc_t any_s_id
        abc_t any_uns_id
        abc_t glide_j
        abc_t glide_w

        ActionSpaceOpt()

    cdef cppclass ActionSpace nogil:
        ActionSpaceOpt opt

cdef extern from "mcts_cpp/env.hpp":
    cdef cppclass EnvOpt nogil:
        VocabIdSeq start_ids
        VocabIdSeq end_ids
        float final_reward
        float step_penalty

        EnvOpt()

    cdef cppclass Env nogil:
        Env(EnvOpt, ActionSpaceOpt, WordSpaceOpt)

        EnvOpt opt
        TreeNode *start
        TreeNode *end

        void register_permissible_change(abc_t, abc_t)
        void register_cl_map(abc_t, abc_t)
        void register_gbj_map(abc_t, abc_t)
        void register_gbw_map(abc_t, abc_t)
        float get_edit_dist(IdSeq, IdSeq)
        TreeNode *apply_action(TreeNode *node, abc_t, abc_t, abc_t, abc_t, abc_t, abc_t, SpecialType) except +

cdef extern from "mcts_cpp/node.hpp":

    ctypedef vector[pair[int, size_t]] Affected
    ctypedef pair[int, abc_t] ChosenChar

    cdef cppclass BaseNode nogil:
        pass

    ctypedef BaseNode * BNptr

    cdef cppclass TreeNode nogil:
        BaseNode *parent
        ChosenChar chosen_char
        bool stopped

        vector[abc_t] permissible_chars
        vector[Affected] affected
        vector[BNptr] children

        vector[float] priors
        vector[visit_t] action_counts
        vector[float] total_values
        visit_t visit_count
        int max_index
        float max_value

        bool is_expanded()
        bool is_evaluated()
        vector[float] get_scores(float)

        vector[Wptr] words
        int depth

        float dist
        bool done

        bool is_leaf()
        TreeNode *play()
        IdSeq get_id_seq(int)
        size_t size()

ctypedef TreeNode * TNptr

cdef extern from "mcts_cpp/mcts.hpp":
    cdef cppclass MctsOpt nogil:
        float puct_c
        int game_count
        float virtual_loss
        int num_threads

        MctsOpt()

    cdef cppclass Mcts nogil:
        MctsOpt opt

        Mcts(Env *, MctsOpt)

        vector[TNptr] select(TreeNode *, int, int)
        void backup(vector[TNptr], vector[float])

# Convertible types between numpy and c++ template.
ctypedef fused convertible:
    int
    float
    long
    abc_t
    bool

cdef inline vector[convertible] np2vector(convertible[::1] arr):
    cdef size_t n = arr.shape[0]
    cdef size_t i
    cdef vector[convertible] vec = vector[convertible](n)
    for i in range(n):
        vec[i] = arr[i]
    return vec

cdef inline vector[vector[convertible]] np2nested(convertible[:, ::1] arr,
                                                  long[::1] lengths):
    cdef size_t n = lengths.shape[0]
    cdef vector[vector[convertible]] ret = vector[vector[convertible]](n)
    cdef vector[convertible] item
    cdef long m
    for i in range(n):
        m = lengths[i]
        item = vector[convertible](m)
        for j in range(m):
            item[j] = arr[i, j]
        ret[i] = item
    return ret
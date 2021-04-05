# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

cdef extern from "mcts_cpp/word.cpp": pass
cdef extern from "mcts_cpp/action.cpp": pass
cdef extern from "mcts_cpp/env.cpp": pass
cdef extern from "mcts_cpp/node.cpp": pass
cdef extern from "mcts_cpp/mcts.cpp": pass
cdef extern from "mcts_cpp/lru_cache.cpp": pass

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

    cdef cppclass PlayStrategy:
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

cdef extern from "mcts_cpp/common.hpp" namespace "PlayStrategy":
    cdef PlayStrategy MAX
    cdef PlayStrategy SAMPLE_AC
    cdef PlayStrategy SAMPLE_MV

cdef extern from "mcts_cpp/word.hpp":
    cdef cppclass Word nogil:
        IdSeq id_seq

    cdef cppclass WordSpaceOpt nogil:
        vector[vector[float]] dist_mat
        float ins_cost
        bool use_alignment
        vector[bool] is_vowel
        vector[bool] is_consonant
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
        int site_threshold
        float dist_threshold

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

        size_t evict(size_t)
        void register_permissible_change(abc_t, abc_t)
        void evaluate(TreeNode *, vector[vector[float]], vector[float])
        void register_cl_map(abc_t, abc_t)
        void register_gbj_map(abc_t, abc_t)
        void register_gbw_map(abc_t, abc_t)
        float get_edit_dist(IdSeq, IdSeq)
        TreeNode *apply_action(TreeNode *, abc_t, abc_t, abc_t, abc_t, abc_t, abc_t, SpecialType) except +
        int get_num_affected(TreeNode *, abc_t, abc_t, abc_t, abc_t, abc_t, abc_t, SpecialType) except +
        void clear_stats(TreeNode *, bool)
        void clear_priors(TreeNode *, bool)
        size_t get_num_words()
        void add_noise(TreeNode *, vector[vector[float]], vector[float], float)
        size_t get_max_end_length()
        vector[vector[abc_t]] expand_all_actions(TreeNode *)

cdef extern from "mcts_cpp/node.hpp":

    ctypedef vector[pair[int, size_t]] Affected
    ctypedef pair[int, abc_t] ChosenChar

    cdef cppclass SelectionOpt nogil:
        float puct_c
        float heur_c
        bool add_noise
        bool use_num_misaligned
        bool use_max_value

        SelectionOpt()

    cdef cppclass BaseNode nogil:
        vector[bool] get_pruned()
        vector[abc_t] get_actions()
        vector[float] get_priors()
        vector[visit_t] get_action_counts()
        vector[float] get_total_values()
        vector[float] get_max_values()
        visit_t get_visit_count()
        bool is_tree_node()
        bool is_transitional()

    cdef cppclass TransitionNode nogil:
        vector[float] get_rewards()

    cdef cppclass TreeNode nogil:
        bool stopped

        vector[Affected] affected

        vector[float] priors
        int max_index
        float max_value

        bool is_expanded()
        bool is_evaluated()
        vector[float] get_scores(float)

        vector[Wptr] words

        float get_dist()
        bool is_done()
        bool is_leaf()
        IdSeq get_id_seq(int)
        size_t size()
        size_t get_num_actions()
        pair[vector[vector[size_t]], vector[vector[size_t]]] get_alignments()

ctypedef TreeNode * TNptr
ctypedef BaseNode * BNptr

cdef extern from "mcts_cpp/mcts.hpp":
    cdef cppclass MctsOpt nogil:
        int game_count
        float virtual_loss
        int num_threads
        SelectionOpt selection_opt

        MctsOpt()

    cdef cppclass Path nogil:
        Path()
        Path(Path)

        int get_depth()
        vector[BNptr] get_all_nodes()
        vector[size_t] get_all_chosen_indices()
        vector[abc_t] get_all_chosen_actions()
        void merge(Path)
        TreeNode *get_last_node()
        vector[abc_t] get_last_action_vec()

    cdef cppclass Mcts nogil:
        MctsOpt opt

        Mcts(Env *, MctsOpt)

        vector[Path] select(TreeNode *, int, int, int)
        vector[Path] select(TreeNode *, int, int, int, Path)
        TreeNode * select_one_pi_step(TreeNode *)
        TreeNode * select_one_random_step(TreeNode *)
        void eval()
        void train()
        void backup(vector[Path], vector[float])
        Path play(TreeNode *, int, PlayStrategy, float)

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
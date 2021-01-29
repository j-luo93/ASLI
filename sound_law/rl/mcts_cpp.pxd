# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "mcts_cpp/timer.cpp": pass
cdef extern from "mcts_cpp/stats.cpp": pass
cdef extern from "mcts_cpp/site.cpp": pass
cdef extern from "mcts_cpp/word.cpp": pass
cdef extern from "mcts_cpp/tree_node.cpp": pass
cdef extern from "mcts_cpp/action.cpp": pass
cdef extern from "mcts_cpp/env.cpp": pass
cdef extern from "mcts_cpp/mcts.cpp": pass

cdef extern from "mcts_cpp/parallel-hashmap/parallel_hashmap/phmap.h"  namespace "phmap" nogil:
    cdef cppclass flat_hash_map[T, U]:
        U& at(const T&)

    cdef cppclass flat_hash_set[T]:
        pass

cdef extern from "mcts_cpp/ctpl.h" namespace "ctpl" nogil:
    cdef cppclass threadpool:
        pass

cdef extern from "mcts_cpp/common.hpp":
    ctypedef unsigned short abc_t
    ctypedef int visit_t
    ctypedef vector[abc_t] IdSeq
    ctypedef vector[IdSeq] VocabIdSeq

    ctypedef unsigned long uai_t
    ctypedef unsigned long usi_t

    cdef cppclass SpecialType:
        pass

    cdef cppclass Stress:
        pass

    ctypedef threadpool Pool

cdef extern from "mcts_cpp/stats.hpp":
    cdef cppclass Stats nogil:
        void enable_timer()
        void disable_timer()
        void show_stats()

    cdef Stats stats

cdef extern from "mcts_cpp/common.hpp" namespace "SpecialType":
    cdef SpecialType CLL
    cdef SpecialType CLR
    cdef SpecialType VS
    cdef SpecialType GBJ
    cdef SpecialType GBW

cdef extern from "mcts_cpp/common.hpp" namespace "Stress":
    cdef Stress NOSTRESS
    cdef Stress STRESSED
    cdef Stress UNSTRESSED

cdef extern from "mcts_cpp/common.hpp" namespace "action":
    uai_t combine(abc_t, abc_t, abc_t, abc_t, abc_t, abc_t)
    uai_t combine_special(abc_t, abc_t, abc_t, abc_t, abc_t, abc_t, SpecialType)
    abc_t get_after_id(uai_t)
    abc_t get_before_id(uai_t)
    abc_t get_d_post_id(uai_t)
    abc_t get_post_id(uai_t)
    abc_t get_d_pre_id(uai_t)
    abc_t get_pre_id(uai_t)

cdef extern from "mcts_cpp/site.hpp":
    cdef cppclass SiteNode nogil:
        usi_t site

    ctypedef SiteNode * SNptr
    cdef cppclass SiteSpace nogil:
        # unordered_map[usi_t, SNptr] nodes
        flat_hash_map[usi_t, SNptr] nodes
        abc_t sot_id
        abc_t eot_id
        abc_t any_id
        abc_t emp_id

        SiteSpace(abc_t, abc_t, abc_t, abc_t, abc_t, abc_t, abc_t)

        size_t size()
        void get_node(SiteNode *, usi_t)
        void get_nodes(Pool *, vector[vector[SNptr]], vector[vector[usi_t]])

    cdef cppclass GraphNode nogil:
        SiteNode *base
        int num_sites
        # unordered_set[int] linked_words
        flat_hash_set[int] linked_words

    ctypedef GraphNode * GNptr
    cdef cppclass SiteGraph nogil:
        # unordered_map[usi_t, GNptr] nodes
        flat_hash_map[usi_t, GNptr] nodes
        void add_root(SiteNode *, int)

cdef extern from "mcts_cpp/word.hpp":
    ctypedef Word * Wptr
    cdef cppclass Word nogil:
        IdSeq id_seq
        flat_hash_map[uai_t, Wptr] neighbors
        # unordered_map[uai_t, Wptr] neighbors
        vector[SNptr] site_roots
        # unordered_map[int, float] dists
        flat_hash_map[int, float] dists
        string str()

    cdef cppclass WordSpace nogil:
        SiteSpace *site_space
        vector[vector[float]] dist_mat
        float ins_cost

        WordSpace(SiteSpace *, vector[vector[float]], float)

        void get_words(Pool *, vector[Wptr], vector[IdSeq])
        size_t size()
        void set_end_words(vector[Wptr])
        float get_edit_dist(IdSeq, IdSeq)

cdef extern from "mcts_cpp/tree_node.hpp":
    cdef cppclass TreeNode nogil:
        ctypedef TreeNode * TNptr

        vector[Wptr] words
        pair[int, uai_t] prev_action
        TreeNode *parent_node
        bool stopped
        bool done
        int depth
        float dist
        vector[uai_t] action_allowed
        vector[float] prior
        vector[visit_t] action_count
        vector[float] total_value
        visit_t visit_count
        # unordered_map[uai_t, TNptr] neighbors
        # unordered_map[uai_t, float] rewards
        flat_hash_map[uai_t, TNptr] neighbors
        flat_hash_map[uai_t, float] rewards
        float max_value
        int max_index
        uai_t max_action_id

        bool is_leaf()
        vector[float] get_scores(float)
        int get_best_i(float)
        void expand(vector[float])
        string str()
        IdSeq get_id_seq(int)
        size_t size()
        size_t get_num_descendants()
        void clear_stats(bool)
        void add_noise(vector[float], float)

    cdef cppclass DetachedTreeNode nogil:
        VocabIdSeq vocab_i
        vector[uai_t] action_allowed

        DetachedTreeNode(TreeNode *)

        IdSeq get_id_seq(int)
        size_t size()

ctypedef TreeNode * TNptr
ctypedef DetachedTreeNode * DTNptr
ctypedef fused anyTNptr:
    TNptr
    DTNptr

cdef extern from "mcts_cpp/action.hpp":
    cdef cppclass ActionSpace nogil:
        SiteSpace *site_space
        WordSpace *word_space
        float dist_threshold
        int site_threshold

        ActionSpace(SiteSpace *, WordSpace *, float, int)

        void register_edge(abc_t, abc_t)
        void register_cl_map(abc_t, abc_t)
        void set_vowel_info(vector[bool], vector[int], vector[Stress]);
        void set_glide_info(abc_t, abc_t)
        void set_action_allowed(Pool *, vector[TNptr])
        void set_action_allowed(TreeNode *)
        IdSeq apply_action(IdSeq, uai_t)
        vector[uai_t] get_similar_actions(uai_t)

cdef extern from "mcts_cpp/env.hpp":
    cdef cppclass Env nogil:
        ActionSpace *action_space
        WordSpace *word_space
        TreeNode *start
        TreeNode *end
        float final_reward
        float step_penalty

        Env(ActionSpace *, WordSpace *, VocabIdSeq, VocabIdSeq, float, float)

        TreeNode *apply_action(TreeNode *, int, uai_t)

cdef extern from "mcts_cpp/mcts.hpp":
    cdef cppclass Mcts nogil:
        Env *env
        float puct_c
        int game_count
        float virtual_loss
        int num_threads

        Mcts(Env *, float, int, float, int)

        vector[TNptr] select(TreeNode *, int, int)
        void backup(vector[TNptr], vector[float])
        uai_t play(TreeNode *)
        void set_logging_options(int, bool)

# Convertible types between numpy and c++ template.
ctypedef fused convertible:
    int
    float
    long
    abc_t
    bool

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

cdef inline vector[convertible] np2vector(convertible[::1] arr):
    cdef size_t n = arr.shape[0]
    cdef size_t i
    cdef vector[convertible] vec = vector[convertible](n)
    for i in range(n):
        vec[i] = arr[i]
    return vec
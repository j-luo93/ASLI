# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.list cimport list as cpp_list
from libcpp.pair cimport pair
from libcpp cimport bool

cdef extern from "TreeNode.cpp": pass
cdef extern from "Action.cpp": pass
cdef extern from "Env.cpp": pass
cdef extern from "Word.cpp": pass
cdef extern from "Site.cpp": pass
cdef extern from "SiteGraph.cpp": pass


cdef extern from "<boost/unordered_map.hpp>" namespace "boost" nogil:
    cdef cppclass unordered_map[T, U, HASH=*, PRED=*, ALLOCATOR=*]:
        size_t size()
        U at(T)


cdef extern from "<boost/unordered_set.hpp>" namespace "boost" nogil:
    cdef cppclass unordered_set[T, HASH=*, PRED=*, ALLOCATOR=*]:
        pass


cdef extern from "<array>" namespace "std" nogil:
    cdef cppclass six "6":
        pass

    cdef cppclass five "5":
        pass

    cdef cppclass cpp_array[T, S, ALLOCATOR=*]:
        T at(size_t)


cdef extern from "common.hpp":
    ctypedef short abc_t
    ctypedef unsigned int action_t
    ctypedef int visit_t
    ctypedef cpp_array[abc_t, six] Action
    ctypedef vector[abc_t] IdSeq
    ctypedef vector[IdSeq] VocabIdSeq
    ctypedef cpp_array[abc_t, five] Site


cdef extern from "Word.hpp":
    ctypedef Word * Wptr
cdef extern from "TreeNode.hpp":
    cdef cppclass TreeNode nogil:
        ctypedef TreeNode * TNptr

        void debug()
        bool is_leaf()
        vector[float] get_scores(float)
        action_t get_best_i(float)
        action_t select(float, int, float)
        void expand(vector[float])
        void backup(float, float, int, float)
        void play()
        cpp_list[pair[action_t, float]] get_path()
        void add_noise(vector[float], float)
        IdSeq get_id_seq(int)
        size_t size()

        vector[Wptr] words
        bool done
        float dist
        vector[action_t] action_allowed
        TreeNode *parent_node
        pair[action_t, action_t] prev_action
        unordered_map[action_t, TNptr] neighbors
        unordered_map[action_t, float] rewards
        vector[float] prior
        vector[visit_t] action_count
        vector[float] total_value
        visit_t visit_count
        float max_value
        int max_index
        action_t max_action_id
        bool played
        void clear_stats(bool)

    cdef cppclass DetachedTreeNode nogil:
        DetachedTreeNode(TreeNode *)

        IdSeq get_id_seq(int)
        size_t size()

        vector[action_t] action_allowed

ctypedef TreeNode * TNptr
ctypedef DetachedTreeNode * DTNptr
ctypedef fused anyTNptr:
    TNptr
    DTNptr


cdef extern from "Action.hpp":
    cdef cppclass ActionSpace nogil:
        ActionSpace(SiteSpace *, WordSpace *)

        SiteSpace *site_space
        WordSpace *word_space

        void register_edge(abc_t, abc_t)
        action_t get_action_id(Action)
        Action get_action(action_t)
        void set_action_allowed(TreeNode *)
        size_t size()
        vector[abc_t] expand_a2i()


cdef extern from "Env.hpp":
    cdef cppclass Env nogil:
        Env(WordSpace *, ActionSpace *, VocabIdSeq, float, float)

        TreeNode *apply_action(TreeNode *, action_t, action_t)

        WordSpace *word_space
        ActionSpace *action_space
        TreeNode *start
        TreeNode *end
        float final_reward
        float step_penalty


cdef extern from "Word.hpp":
    ctypedef SiteNode * SNptr
    cdef cppclass Word nogil:

        size_t size()
        void debug()

        IdSeq id_seq
        float dist
        bool done
        unordered_map[Action, Wptr] neighbors
        vector[SNptr] site_roots

    cdef cppclass WordSpace nogil:
        WordSpace(SiteSpace *, vector[vector[float]], float, VocabIdSeq)

        SiteSpace *site_space
        vector[vector[float]] dist_mat
        float ins_cost
        vector[Wptr] end_words

        Word *get_word(IdSeq, int, bool)
        Word *apply_action(Word *, Action, int)
        size_t size()


cdef extern from "Site.hpp":
    cdef cppclass SiteNode nogil:

        SiteNode(Site)

        void debug()

        Site site
        SiteNode *lchild
        SiteNode *rchild

    cdef cppclass SiteSpace nogil:
        SiteNode *get_node(abc_t, abc_t, abc_t, abc_t, abc_t)


cdef extern from "SiteGraph.hpp":
    cdef cppclass GraphNode nogil:
        SiteNode *base
        GraphNode *lchild
        GraphNode *rchild
        int num_sites
        unordered_set[int] linked_words

    cdef cppclass SiteGraph nogil:
        ctypedef GraphNode * GNptr
        SiteGraph(SiteSpace *)

        void *add_root(SiteNode *, int)

        SiteSpace *site_space
        unordered_map[Site, GNptr] nodes


cdef inline VocabIdSeq np2vocab(long[:, ::1] arr,
                                long[::1] lengths,
                                size_t n):
    cdef size_t i, j, m
    cdef VocabIdSeq vocab = VocabIdSeq(n)
    cdef IdSeq id_seq
    for i in range(n):
        m = lengths[i]
        id_seq = IdSeq(m)
        for j in range(m):
            id_seq[j] = arr[i, j]
        vocab[i] = id_seq
    return vocab


# Convertible types between numpy and c++ template.
ctypedef fused convertible:
    float
cdef inline vector[convertible] np2vector(convertible[::1] arr, size_t n):
    cdef size_t i
    cdef vector[convertible] vec = vector[convertible](n)
    for i in range(n):
        vec[i] = arr[i]
    return vec
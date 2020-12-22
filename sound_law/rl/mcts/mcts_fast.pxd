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


cdef extern from "robin_hood.h" namespace "robin_hood" nogil:
    cdef cppclass unordered_map[T, U, HASH=*, PRED=*, ALLOCATOR=*]:
        unordered_map() except +
        size_t size()
        U& at(const T&)
        U& operator[](T&)
        cppclass iterator:
            pair[T, U]& operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        iterator find(T&)
        iterator begin()
        iterator end()

    cdef cppclass unordered_set[T, HASH=*, PRED=*, ALLOCATOR=*]:
        unordered_set() except +
        cppclass iterator:
            T& operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        iterator find(T&)
        iterator begin()
        iterator end()
        pair[iterator, bint] insert(T&)
        size_t size()


cdef extern from "common.hpp":
    ctypedef unsigned short abc_t
    ctypedef int visit_t
    ctypedef unsigned long tn_cnt_t
    ctypedef vector[abc_t] IdSeq
    ctypedef vector[IdSeq] VocabIdSeq
    ctypedef unsigned long uai_t
    ctypedef unsigned long usi_t
cdef extern from "common.hpp" namespace "action":
    uai_t combine(abc_t, abc_t, abc_t, abc_t, abc_t, abc_t)
    abc_t get_after_id(uai_t)
    abc_t get_before_id(uai_t)
    abc_t get_d_post_id(uai_t)
    abc_t get_post_id(uai_t)
    abc_t get_d_pre_id(uai_t)
    abc_t get_pre_id(uai_t)


cdef extern from "Word.hpp":
    # Use different name to avoid redeclaration.
    ctypedef Word * fwd_Wptr
cdef extern from "TreeNode.hpp":
    cdef cppclass TreeNode nogil:
        ctypedef TreeNode * TNptr

        void debug()
        bool is_leaf()
        vector[float] get_scores(float)
        int get_best_i(float)
        int select(float, int, float)

        void expand(vector[float])
        void backup(float, float, int, float)
        void play()
        cpp_list[pair[uai_t, float]] get_path()
        void add_noise(vector[float], float)
        IdSeq get_id_seq(int)
        void clear_stats(bool)
        size_t size()
        size_t get_num_descendants()
        size_t clear_cache(float)

        tn_cnt_t idx
        vector[fwd_Wptr] words
        bool stopped
        bool done
        float dist
        vector[uai_t] action_allowed
        TreeNode *parent_node
        pair[int, uai_t] prev_action
        unordered_map[uai_t, TNptr] neighbors
        unordered_map[uai_t, float] rewards
        vector[float] prior
        vector[visit_t] action_count
        vector[float] total_value
        visit_t visit_count
        float max_value
        int max_index
        uai_t max_action_id
        bool played

    cdef cppclass DetachedTreeNode nogil:
        DetachedTreeNode(TreeNode *)

        IdSeq get_id_seq(int)
        size_t size()

        vector[uai_t] action_allowed

ctypedef TreeNode * TNptr
ctypedef DetachedTreeNode * DTNptr
ctypedef fused anyTNptr:
    TNptr
    DTNptr


cdef extern from "Action.hpp":
    cdef cppclass ActionSpace nogil:
        ActionSpace(SiteSpace *, WordSpace *, float, int)

        SiteSpace *site_space
        WordSpace *word_space
        int num_threads

        void register_edge(abc_t, abc_t)
        void set_action_allowed(TreeNode *)
        void set_action_allowed(vector[TNptr])


cdef extern from "Env.hpp":
    cdef cppclass Env nogil:
        Env(WordSpace *, ActionSpace *, VocabIdSeq, float, float)

        TreeNode *apply_action(TreeNode *, int, uai_t)

        WordSpace *word_space
        ActionSpace *action_space
        TreeNode *start
        TreeNode *end
        float final_reward
        float step_penalty


cdef extern from "Word.hpp":
    ctypedef SiteNode * SNptr
    cdef cppclass Word nogil:
        # This does not seem to trigger redeclaration error.
        ctypedef Word * Wptr

        size_t size()
        void debug()

        IdSeq id_seq
        float dist
        bool done
        unordered_map[uai_t, Wptr] neighbors
        vector[SNptr] site_roots

    cdef cppclass WordSpace nogil:
        ctypedef Word * Wptr
        WordSpace(SiteSpace *, vector[vector[float]], float, VocabIdSeq)

        SiteSpace *site_space
        vector[vector[float]] dist_mat
        float ins_cost
        vector[Wptr] end_words

        Word *get_word(IdSeq, int, bool)
        Word *apply_action(Word *, uai_t, int)
        Word *apply_action_no_lock(Word *, uai_t, int)
        size_t size()
# Only the module-level ctypedef can be imported by pyx file.
ctypedef Word * Wptr


cdef extern from "Site.hpp":
    cdef cppclass SiteNode nogil:
        SiteNode(usi_t)

        void debug()

        usi_t site
        SiteNode *lchild
        SiteNode *rchild

    cdef cppclass SiteSpace nogil:
        SiteSpace(abc_t, abc_t, abc_t, abc_t)
        SiteNode *get_node(abc_t, abc_t, abc_t, abc_t, abc_t)
        size_t size()


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
        unordered_map[usi_t, GNptr] nodes


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
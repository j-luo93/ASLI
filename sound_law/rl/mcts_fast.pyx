# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref
from cython.parallel import prange
from libcpp cimport bool

cdef extern from "TreeNode.cpp":
    pass

cdef extern from "Action.cpp":
    pass

cdef extern from "Env.cpp":
    pass


cdef extern from "TreeNode.h":
    ctypedef unsigned int uint
    ctypedef vector[uint] IdSeq
    ctypedef vector[IdSeq] VocabIdSeq
    cdef cppclass TreeNode nogil:
        TreeNode(VocabIdSeq, TreeNode *) except +

        void add_edge(uint, TreeNode *)
        bool has_acted(uint)
        uint size()
        void lock()
        void unlock()

        VocabIdSeq vocab_i
        unsigned long dist_to_end
        unordered_map[uint, TreeNode *] edges

cdef extern from "Action.h":
    cdef cppclass Action nogil:
        Action(uint, uint, uint)
        uint action_id
        uint before_id
        uint after_id


cdef extern from "Env.h":
    cdef cppclass Env nogil:
        Env(TreeNode *, TreeNode *) except +

        TreeNode *step(TreeNode *, Action *)

        TreeNode *init_node
        TreeNode *end_node


cdef inline TreeNode* create_node(object lst, TreeNode *end_node=NULL):
    cdef uint n = len(lst)
    cdef uint i, m
    # Initialize vector
    cdef VocabIdSeq vocab_i = VocabIdSeq(n)
    for i in range(n):
        vocab_i[i] = lst[i]

    cdef TreeNode *node = new TreeNode(vocab_i, end_node)
    return node

cdef extern from "unistd.h" nogil:
    unsigned int sleep(unsigned int seconds)

ctypedef TreeNode * TNptr
ctypedef Action * Aptr

cpdef bool test(object lst1, object lst2):
    cdef TreeNode *end = create_node(lst2)
    cdef TreeNode *start = create_node(lst1, end)
    cdef Env *env = new Env(start, end)
    cdef Action *action0 = new Action(0, 2, 22)
    cdef Action *action1 = new Action(1, 2, 2222)
    cdef TreeNode *new_node = env.step(start, action0)
    cdef vector[TNptr] queue = vector[TNptr](2)
    queue[0] = start
    queue[1] = start
    cdef vector[Aptr] action_queue = vector[Aptr](2)
    action_queue[0] = action0
    action_queue[1] = action1
    cdef uint i
    cdef TreeNode *node

    import time; s = time.time()
    with nogil:
        for i in prange(2, num_threads=2):
            node = queue[i]
            action = action_queue[i]
            # node.lock()
            sleep(1)
            # start.unlock()
    print(time.time() - s)
    return start.has_acted(0)




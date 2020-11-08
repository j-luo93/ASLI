# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref
from cython.parallel import prange
from libcpp cimport nullptr
from typing import List

from libcpp cimport bool
import numpy as np
cimport numpy as np

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
        TreeNode(VocabIdSeq) except +
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

cdef inline VocabIdSeq np2vocab(uint[:, ::1] arr, uint n, uint m) except *:
    cdef uint i, j
    cdef VocabIdSeq vocab_i = VocabIdSeq(n)
    cdef IdSeq id_seq
    for i in range(n):
        id_seq = IdSeq(m)
        for j in range(m):
            id_seq[j] = arr[i, j]
        vocab_i[i] = id_seq
    return vocab_i

cdef inline uint[:, ::1] vocab2np(VocabIdSeq vocab_i) except *:
    cdef uint n = vocab_i.size()
    cdef uint m = 0
    # Find the longest sequence.
    cdef uint i, j
    for i in range(n):
        m = max(m, vocab_i[i].size())
    arr = np.zeros([n, m], dtype='uintc')
    cdef uint[:, ::1] arr_view = arr
    cdef IdSeq id_seq
    for i in range(n):
        id_seq = vocab_i[i]
        for j in range(m):
            arr[i, j] = id_seq[j]
    return arr

cdef extern from "unistd.h" nogil:
    unsigned int sleep(unsigned int seconds)

ctypedef TreeNode * TNptr
ctypedef Action * Aptr

cdef class PyTreeNode:
    cdef TNptr ptr

    def __dealloc__(self):
        # Don't free the memory. Just delete the attribute.
        del self.ptr

    @staticmethod
    cdef PyTreeNode from_ptr(TreeNode *ptr):
        cdef PyTreeNode py_tn = PyTreeNode.__new__(PyTreeNode)
        py_tn.ptr = ptr
        return py_tn

    @staticmethod
    cdef PyTreeNode from_np(object arr, PyTreeNode end_node = None):
        cdef uint[:, ::1] arr_view = arr
        cdef uint n = arr.shape[0]
        cdef uint m = arr.shape[1]
        cdef VocabIdSeq vocab_i = np2vocab(arr_view, n, m)
        cdef TreeNode *ptr
        if end_node is None:
            ptr = new TreeNode(vocab_i)
        else:
            ptr = new TreeNode(vocab_i, end_node.ptr)
        return PyTreeNode.from_ptr(ptr)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float np_sum(float[::1] x, uint size) nogil:
    cdef float s = 0.0
    cdef uint i
    for i in range(size):
        s = s + x[i]
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bool test(uint[:, ::1] arr1, uint[:, ::1] arr2):
    cdef PyTreeNode end = PyTreeNode.from_np(arr2)
    cdef PyTreeNode start = PyTreeNode.from_np(arr1, end)
    cdef Env *env = new Env(start.ptr, end.ptr)
    cdef Action *action0 = new Action(0, 2, 22)
    cdef Action *action1 = new Action(1, 2, 2222)
    cdef TreeNode *new_node = env.step(start.ptr, action0)
    cdef vector[TNptr] queue = vector[TNptr](2)
    queue[0] = start.ptr
    queue[1] = start.ptr
    cdef vector[Aptr] action_queue = vector[Aptr](2)
    action_queue[0] = action0
    action_queue[1] = action1
    cdef uint i
    cdef TreeNode *node

    cdef float[::1] Psa_1 = np.random.randn(5).astype('float32')
    cdef float[::1] Psa_2 = np.random.randn(5).astype('float32')
    cdef float[::1] Psa
    cdef float s = 0.0
    cdef uint num_actions = 5

    with nogil:
        for i in prange(2, num_threads=2):
            node = queue[i]
            action = action_queue[i]
            node.lock()
            s += np_sum(Psa_1, num_actions)
            sleep(1)
            node.unlock()
    print(s)
    print(np.asarray(Psa_1).sum()  * 2)
    return start.has_acted(0)




# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref
from cython.parallel import prange
from libcpp cimport nullptr
from typing import List
from libc.stdlib cimport free
from libc.stdio cimport printf

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
    ctypedef vector[long] IdSeq
    ctypedef vector[IdSeq] VocabIdSeq
    cdef cppclass TreeNode nogil:
        TreeNode(VocabIdSeq) except +
        TreeNode(VocabIdSeq, TreeNode *) except +

        void add_edge(long, TreeNode *)
        bool has_acted(long)
        long size()
        void lock()
        void unlock()
        bool is_leaf()

        VocabIdSeq vocab_i
        unsigned long dist_to_end
        unordered_map[long, TreeNode *] edges

cdef extern from "Action.h":
    cdef cppclass Action nogil:
        Action(long, long, long)
        long action_id
        long before_id
        long after_id


cdef extern from "Env.h":
    cdef cppclass Env nogil:
        Env(TreeNode *, TreeNode *) except +

        TreeNode *step(TreeNode *, Action *)

        TreeNode *init_node
        TreeNode *end_node

cdef inline VocabIdSeq np2vocab(long[:, ::1] arr, long n, long m) except *:
    cdef long i, j
    cdef VocabIdSeq vocab_i = VocabIdSeq(n)
    cdef IdSeq id_seq
    for i in range(n):
        id_seq = IdSeq(m)
        for j in range(m):
            id_seq[j] = arr[i, j]
        vocab_i[i] = id_seq
    return vocab_i

cdef inline long[:, ::1] vocab2np(VocabIdSeq vocab_i) except *:
    cdef long n = vocab_i.size()
    cdef long m = 0
    # Find the longest sequence.
    cdef long i, j
    for i in range(n):
        m = max(m, vocab_i[i].size())
    arr = np.zeros([n, m], dtype='long')
    cdef long[:, ::1] arr_view = arr
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
        # # Don't free the memory. Just delete the attribute.
        # del self.ptr
        # free(self.ptr)
        self.ptr = NULL

    @staticmethod
    cdef PyTreeNode from_ptr(TreeNode *ptr):
        cdef PyTreeNode py_tn = PyTreeNode.__new__(PyTreeNode)
        py_tn.ptr = ptr
        return py_tn

    @staticmethod
    cdef PyTreeNode from_np(object arr, PyTreeNode end_node = None):
        cdef long[:, ::1] arr_view = arr
        cdef long n = arr.shape[0]
        cdef long m = arr.shape[1]
        cdef VocabIdSeq vocab_i = np2vocab(arr_view, n, m)
        cdef TreeNode *ptr
        if end_node is None:
            ptr = new TreeNode(vocab_i)
        else:
            ptr = new TreeNode(vocab_i, end_node.ptr)
        return PyTreeNode.from_ptr(ptr)

    @property
    def vocab(self):
        return vocab2np(self.ptr.vocab_i)

    def __str__(self):
        out = list()
        for i in range(self.ptr.vocab_i.size()):
            out.append(' '.join(map(str, self.ptr.vocab_i[i])))
        return '\n'.join(out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float np_sum(float[::1] x, long size) nogil:
    cdef float s = 0.0
    cdef long i
    for i in range(size):
        s = s + x[i]
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline long np_argmax(float[::1] x, long size) nogil:
    cdef long best_i, i
    best_i = 0
    cdef float best_v = x[0]
    for i in range(1, size):
        if best_v < x[i]:
            best_v = x[i]
            best_i = i
    return best_i

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bool test(long[:, ::1] arr1, long[:, ::1] arr2):
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
    cdef long i
    cdef TreeNode *node

    cdef float[::1] Psa_1 = np.random.randn(5).astype('float32')
    cdef float[::1] Psa_2 = np.random.randn(5).astype('float32')
    cdef float[::1] Psa
    cdef float s = 0.0
    cdef long num_actions = 5

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

cpdef object parallel_select(long[:, ::1] init_arr,
                             long[:, ::1] end_arr,
                             long num_sims,
                             long num_threads,
                             long depth_limit):
    cdef PyTreeNode py_end = PyTreeNode.from_np(end_arr)
    cdef TreeNode *end = py_end.ptr
    cdef PyTreeNode py_start = PyTreeNode.from_np(init_arr, py_end)
    cdef TreeNode *start = py_start.ptr
    cdef Env * env = new Env(start, end)

    cdef TreeNode *node, *next_node
    cdef long n_steps_left, i, action_id, num_actions
    Psa = np.random.randn(5).astype('float32')
    cdef float[::1] Psa_view = Psa
    print(Psa)
    print(np_argmax(Psa_view, 5))
    cdef Action *action
    cdef vector[TNptr] selected = vector[TNptr](num_sims)
    for i in range(5):
        action = new Action(i, i, i + 10)
        # j_luo(FIXME) We must make sure that before calling `step`, `start` has an end_node -- see Env.cpp, line 18.
        node = env.step(start, action)
        start.add_edge(i, node)

    with nogil:
        for i in range(num_sims):#, num_threads=num_threads):
            node = start
            n_steps_left = depth_limit
            num_actions = 5
            with gil:
                print(np.asarray(Psa_view))
            while n_steps_left > 0 and node.vocab_i != end.vocab_i and not node.is_leaf():
                action_id = np_argmax(Psa_view, num_actions)
                with gil:
                    print(action_id)
                action = new Action(action_id, action_id, action_id + 10)
                next_node = env.step(node, action)
                n_steps_left = n_steps_left - 1
                node = next_node
            selected[i] = node
    return [PyTreeNode.from_ptr(ptr) for ptr in selected]

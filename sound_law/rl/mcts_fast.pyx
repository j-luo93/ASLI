# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref
from cython.parallel import prange
from libcpp cimport nullptr
from typing import List
from libc.stdio cimport printf

from libcpp cimport bool
import numpy as np
cimport numpy as np

from sound_law.data.alphabet import PAD_ID

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
        long get_best_action_id(float)
        void expand(vector[float], vector[bool])
        void virtual_backup(long, long, float)
        void backup(float, long, float)

        VocabIdSeq vocab_i
        long dist_to_end
        long prev_action
        unordered_map[long, TreeNode *] edges
        vector[float] prior
        vector[long] action_count
        long visit_count
        vector[float] total_value

cdef extern from "Action.h":
    cdef cppclass Action nogil:
        Action(long, long, long)
        long action_id
        long before_id
        long after_id
    cdef cppclass ActionSpace nogil:
        ActionSpace()
        void register_action(long, long)
        Action *get_action(long)
        vector[bool] get_action_mask(TreeNode *)
        long size()


cdef extern from "Env.h":
    cdef cppclass Env nogil:
        Env(TreeNode *, TreeNode *) except +

        TreeNode *step(TreeNode *, Action *) except +

        TreeNode *init_node
        TreeNode *end_node

cdef inline VocabIdSeq np2vocab(long[:, ::1] arr,
                                long[::1] lengths,
                                long n) except *:
    cdef long i, j, m
    cdef VocabIdSeq vocab_i = VocabIdSeq(n)
    cdef IdSeq id_seq
    for i in range(n):
        m = lengths[i]
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
    arr = np.full([n, m], PAD_ID, dtype='long')
    cdef long[:, ::1] arr_view = arr
    cdef IdSeq id_seq
    for i in range(n):
        id_seq = vocab_i[i]
        for j in range(id_seq.size()):
            arr_view[i, j] = id_seq[j]
    return arr

# Convertible types between numpy and c++ template.
ctypedef fused convertible:
    float
    bool

cdef inline vector[convertible] np2vector(convertible[::1] arr, long n) except *:
    cdef long i
    cdef vector[convertible] vec = vector[convertible](n)
    for i in range(n):
        vec[i] = arr[i]
    return vec

cdef extern from "unistd.h" nogil:
    unsigned int sleep(unsigned int seconds)

ctypedef TreeNode * TNptr
ctypedef Action * Aptr

cdef class PyTreeNode:
    cdef TNptr ptr
    cdef public PyTreeNode end_node

    def __dealloc__(self):
        # # Don't free the memory. Just delete the attribute.
        # del self.ptr
        # free(self.ptr)
        # FIXME(j_luo) Make sure this is correct
        self.ptr = NULL

    def __cinit__(self,
                  *args,
                  object arr = None,
                  object lengths = None,
                  PyTreeNode end_node = None,
                  bool from_ptr = False,
                  **kwargs):
        """`arr` is converted to `vocab_i`, which is then used to construct a c++ TreeNode object. Use this for creating PyTreeNode in python."""
        # Skip creating a new c++ TreeNode object since it would be handled by `from_ptr` instead.
        if arr is None or lengths is None:
            assert from_ptr, 'You must either construct using `from_ptr` or provide `arr` and `lengths` here.'
            return

        cdef long[:, ::1] arr_view = arr
        cdef long n = arr.shape[0]
        assert n == lengths.shape[0], '`arr` and `lengths` must have the same length.'
        cdef long[::1] lengths_view = lengths

        cdef VocabIdSeq vocab_i = np2vocab(arr_view, lengths_view, n)
        if end_node is None:
            self.ptr = new TreeNode(vocab_i)
        else:
            self.ptr = new TreeNode(vocab_i, end_node.ptr)
        self.end_node = end_node

    # @staticmethod
    # cdef PyTreeNode from_ptr(TreeNode *ptr):
    #     """This is used in cython code to wrap around a c++ TreeNode object."""
    #     cdef PyTreeNode py_tn = PyTreeNode.__new__(PyTreeNode, from_ptr=True)
    #     py_tn.ptr = ptr
    #     return py_tn
    def __len__(self):
        return self.ptr.size()

    @property
    def vocab_array(self):
        return np.asarray(vocab2np(self.ptr.vocab_i))

    @property
    def vocab(self):
        return self.ptr.vocab_i

    @property
    def prior(self):
        return np.asarray(self.ptr.prior)

    def __str__(self):
        out = f'visit_count: {self.ptr.visit_count}\n'
        out += f'action_count:\n'
        out += '[' + ', '.join(map(str, self.ptr.action_count)) + ']\n'
        out += f'action_prior:\n'
        out += '[' + ', '.join([f'{p:.3f}' for p in self.ptr.prior]) + ']\n'
        out += f'total_value:\n'
        out += '[' + ', '.join([f'{v:.3f}' for v in self.ptr.total_value]) + ']\n'
        vocab = list()
        for i in range(self.ptr.vocab_i.size()):
            vocab.append(' '.join(map(str, self.ptr.vocab_i[i])))
        out += '\n'.join(vocab)
        return out

    def expand(self, float[::1] prior, bool[::1] action_mask):
        cdef long n = prior.shape[0]
        cdef vector[float] prior_vec = np2vector(prior, n)
        cdef vector[bool] action_mask_vec = np2vector(action_mask, n)
        self.ptr.expand(prior_vec, action_mask_vec)

    def backup(self, float value, long game_count, float virtual_loss):
        self.ptr.backup(value, game_count, virtual_loss)

    @property
    def prev_action(self):
        return self.ptr.prev_action

    @property
    def dist_to_end(self):
        return self.ptr.dist_to_end


cdef class PyAction:
    """This is a wrapper class for c++ class Action. It should be created by a PyActionSpace object with registered actions."""
    cdef Aptr ptr

    def __cinit__(self, *args, bool from_ptr=False, **kwargs):
        assert from_ptr, 'You should only create this object by calling `from_ptr`.'

    def __dealloc__(self):
        self.ptr = NULL
        # del self.ptr

    # @staticmethod
    # cdef PyAction from_ptr(cls, Action *ptr):
    #     cdef PyAction py_a = cls.__new__(cls, from_ptr=True)
    #     py_a.ptr = ptr
    #     return py_a
    @property
    def action_id(self):
        return self.ptr.action_id

    @property
    def before_id(self):
        return self.ptr.before_id

    @property
    def after_id(self):
        return self.ptr.after_id

# NOTE(j_luo) Using staticmethod as the tutorial suggests doesn't work as a flexible factory method -- you might want to control the `cls` in case of subclassing it.
cdef PyAction wrap_action(cls, Action *ptr):
    cdef PyAction py_a = cls.__new__(cls, from_ptr=True)
    py_a.ptr = ptr
    return py_a

cdef PyTreeNode wrap_node(cls, TreeNode *ptr):
    cdef PyTreeNode py_tn = cls.__new__(cls, from_ptr=True)
    py_tn.ptr = ptr
    return py_tn

ctypedef ActionSpace * ASptr

cdef class PyActionSpace:
    cdef ASptr ptr
    action_cls = PyAction

    def __cinit__(self):
        self.ptr = new ActionSpace()

    def __dealloc__(self):
        self.ptr = NULL
        # del self.ptr

    def register_action(self, long before_id, long after_id):
        self.ptr.register_action(before_id, after_id)

    def get_action_mask(self, PyTreeNode py_node):
        cdef vector[bool] action_mask = self.ptr.get_action_mask(py_node.ptr)
        return np.asarray(action_mask)

    def get_action(self, long action_id):
        cdef Action *action = self.ptr.get_action(action_id)
        action_cls = type(self).action_cls
        return wrap_action(action_cls, action)

    def __len__(self):
        return self.ptr.size()

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_action(i)

ctypedef Env * Envptr

cdef class PyEnv:
    cdef Envptr ptr

    def __cinit__(self, PyTreeNode init_node, PyTreeNode end_node, *args, **kwargs):
        self.ptr = new Env(init_node.ptr, end_node.ptr)

    def __dealloc__(self):
        self.ptr = NULL
        # del self.ptr

    def step(self, PyTreeNode node, PyAction action):
        cdef TreeNode *next_node = self.ptr.step(node.ptr, action.ptr)
        return wrap_node(type(node), next_node)

    def is_done(self, PyTreeNode node):
        return self.ptr.end_node == node.ptr


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef inline float np_sum(float[::1] x, long size) nogil:
#     cdef float s = 0.0
#     cdef long i
#     for i in range(size):
#         s = s + x[i]
#     return s

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef inline long np_argmax(float[::1] x, long size) nogil:
#     cdef long best_i, i
#     best_i = 0
#     cdef float best_v = x[0]
#     for i in range(1, size):
#         if best_v < x[i]:
#             best_v = x[i]
#             best_i = i
#     return best_i

# cdef inline long vector_argmax(vector[float] vec) nogil:
#     cdef long best_i, i
#     best_i = 0
#     cdef float best_v = vec[0]
#     for i in range(1, vec.size()):
#         if best_v < vec[i]:
#             best_v = vec[i]
#             best_i = i
#     return best_i


# # FIXME(j_luo) rename node to state?
cpdef object parallel_select(PyTreeNode py_root,
                             PyTreeNode py_end,
                             PyActionSpace py_as,
                             PyEnv py_env,
                             long num_sims,
                             long num_threads,
                             long depth_limit,
                             float puct_c,
                             long game_count,
                             float virtual_loss):
    cdef TreeNode *end = py_end.ptr
    cdef TreeNode *root = py_root.ptr
    cdef Env *env = py_env.ptr

    cdef TreeNode *node, *next_node
    cdef long n_steps_left, i, action_id
    cdef Action *action
    cdef vector[TNptr] selected = vector[TNptr](num_sims)
    cdef ActionSpace *action_space = py_as.ptr

    with nogil:
        for i in prange(num_sims, num_threads=num_threads):
            node = root
            n_steps_left = depth_limit
            while n_steps_left > 0 and not node.is_leaf() and node.vocab_i != end.vocab_i:
                node.lock()
                action_id = node.get_best_action_id(puct_c)
                action = action_space.get_action(action_id)
                next_node = env.step(node, action)
                n_steps_left = n_steps_left - 1
                node.virtual_backup(action_id, game_count, virtual_loss)
                node.unlock()

                node = next_node
            selected[i] = node
    tn_cls = type(py_root)
    return [wrap_node(tn_cls, ptr) for ptr in selected]
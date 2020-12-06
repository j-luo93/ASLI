# distutils: language = c++
from .mcts_fast cimport SiteSpace, VocabIdSeq, np2vocab
import numpy as np
cimport numpy as np


cdef class PySiteSpace:
    cdef SiteSpace *ptr

    def __cinit__(self):
        self.ptr = new SiteSpace()

    def __dealloc__(self):
        del self.ptr


cdef class PyWordSpace:
    cdef WordSpace *ptr

    def __cinit__(self, PySiteSpace py_ss, np_dist_mat, float ins_cost, arr, lengths):
        # Convert numpy dist_mat to vector.
        cdef float[:, ::1] dist_mat_view = np_dist_mat
        cdef size_t n = np_dist_mat.shape[0]
        cdef size_t m = np_dist_mat.shape[1]
        cdef vector[vector[float]] dist_mat = vector[vector[float]](n)
        cdef size_t i, j
        cdef vector[float] vec
        for i in range(n):
            vec = vector[float](m)
            for j in range(m):
                vec[j] = dist_mat_view[i, j]
            dist_mat[i] = vec

        # Obtain vocab
        cdef VocabIdSeq vocab = np2vocab(arr, lengths, len(arr))
        self.ptr = new WordSpace(py_ss.ptr, dist_mat, ins_cost, vocab)

    def __dealloc__(self):
        del self.ptr


cdef class PyActionSpace:
    cdef ActionSpace *ptr

    def __cinit__(self, PySiteSpace py_ss, PyWordSpace py_ws):
        self.ptr = new ActionSpace(py_ss.ptr, py_ws.ptr)

    def __dealloc__(self):
        del self.ptr


cdef class PyEnv:
    cdef Env *ptr

    def __cinit__(self, PyWordSpace py_ws, PyActionSpace py_as, arr, lengths, float final_reward, float step_penalty):
        cdef VocabIdSeq vocab = np2vocab(arr, lengths, len(arr))
        self.ptr = new Env(py_ws.ptr, py_as.ptr, vocab, final_reward, step_penalty)

    def __dealloc__(self):
        del self.ptr


cpdef test_env():
    cdef PySiteSpace py_ss = PySiteSpace()
    dist_mat = np.random.randn(3, 3).astype('float32')
    s_arr = np.random.randint(3, size=[10, 10])
    s_lengths = np.random.randint(10, size=[10]) + 1
    e_arr = np.random.randint(3, size=[10, 10])
    e_lengths = np.random.randint(10, size=[10]) + 1
    cdef PyWordSpace py_ws = PyWordSpace(py_ss, dist_mat, 1.0, s_arr, s_lengths)
    cdef PyActionSpace py_as = PyActionSpace(py_ss, py_ws)
    cdef PyEnv py_env = PyEnv(py_ws, py_as, e_arr, e_lengths, 1.0, 0.02)

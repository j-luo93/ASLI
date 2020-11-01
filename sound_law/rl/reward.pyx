from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

cpdef object get_rtgs(float[::1] rewards, float discount):
    cdef int n = len(rewards)
    rtgs = np.zeros([n], dtype='float32')
    cdef float[::1] rtgs_view = rtgs
    cdef float accum = 0.0
    for i in range(n -1, -1, -1):
        accum = rewards[i] + accum * discount
        rtgs_view[i] = accum
    return rtgs

cpdef object get_rtgs_list(object rewards, float discount):
    cdef int n = len(rewards)
    ret = list()
    for i in range(n):
        ret.append(get_rtgs(rewards[i], discount))
    return ret

cpdef object get_rtgs_dense(float[::1] rewards, int[::1] offsets, float discount):
    cdef int nr = len(rewards)
    cdef int no = len(offsets)
    cdef int[::1] offsets_view = offsets
    rtgs = np.zeros([nr], dtype='float32')
    cdef float[::1] rtgs_view = rtgs
    cdef int *starts = <int *> malloc(no * sizeof(int))
    cdef int start = 0
    cdef int end = 0
    cdef int i = 0
    cdef int j = 0
    cdef float accum = 0.0

    if offsets_view[-1] != nr:
        raise ValueError(f'The last value for offsets should be equal to the length of rewards.')

    starts[0] = 0
    for i in range(1, no):
        starts[i] = offsets_view[i - 1]

    for i in range(no):
        start = starts[i]
        end = offsets_view[i]
        accum = 0
        for j in range(end - 1, start - 1, -1):
            accum = rewards[j] + accum * discount
            rtgs_view[j] = accum

    free(starts)

    return rtgs
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


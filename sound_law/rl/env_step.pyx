# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np
cimport numpy as np

cpdef inline object _env_step_word(object units, str before, str after):
    """Take a step for a word."""
    return [after if u == before else u for u in units]


cpdef object env_step_words(object lst_units, str before, str after):
    return [_env_step_word(units, before, after) for units in lst_units]


cpdef object env_step_ids(object ids, long before_id, long after_id):
    """Take a step for an id matrix."""
    cdef unsigned int n = ids.shape[0]
    cdef unsigned int m = ids.shape[1]
    cdef long c_before_id = before_id
    cdef long c_after_id = after_id

    ret = ids.copy()
    cdef long[:, ::1] ret_view = ret
    cdef long[:, ::1] ids_view = ids

    cdef unsigned int i, j
    for i in range(n):
        for j in range(m):
            if ids_view[i, j] == c_before_id:
                ret_view[i, j] = c_after_id

    return ret
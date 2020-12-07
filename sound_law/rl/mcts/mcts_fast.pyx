# distutils: language = c++
from .mcts_fast cimport SiteSpace, VocabIdSeq, np2vocab, action_t, Action, TNptr, np2vector, anyTNptr
from libcpp cimport nullptr
import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython


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

    def register_edge(self, abc_t before_id, abc_t after_id):
        self.ptr.register_edge(before_id, after_id)

    def set_action_allowed(self, PyTreeNode node):
        self.ptr.set_action_allowed(node.ptr)


cdef class PyTreeNode:
    cdef TreeNode *ptr

    def __cinit__(self, *, from_ptr=False):
        if not from_ptr:
            raise TypeError(f'Cannot create a PyTreeNode object unless directly wrapping around a TreeNode pointer.')

    @property
    def done(self):
        return self.ptr.done

    def expand(self, prior):
        cdef size_t n = len(prior)
        cdef vector[float] prior_vec = np2vector(prior, n)
        self.ptr.expand(prior_vec)

    @property
    def num_actions(self):
        return self.ptr.action_allowed.size()

    # def detach(self):
    #     cdef DetachedTreeNode *ptr = new DetachedTreeNode(self.ptr)
    #     cdef PyDetachedTreeNode py_dtn = PyDetachedTreeNode.__new__(PyDetachedTreeNode)
    #     py_dtn.ptr = ptr
    #     return py_dtn


cdef class PyDetachedTreeNode:
    cdef DetachedTreeNode *ptr

    def __cinit__(self, *, from_ptr=False):
        if not from_ptr:
            raise TypeError(f'Cannot create a PyDetachedTreeNode object unless directly wrapping around a DetachedTreeNode pointer.')

    def __dealloc__(self):
        del self.ptr


cdef class PyEnv:
    cdef Env *ptr

    cdef public PyWordSpace word_space
    cdef public PyActionSpace action_space
    cdef public PyTreeNode start
    cdef public PyTreeNode end

    tn_cls = PyTreeNode

    def __cinit__(self, PyWordSpace py_ws, PyActionSpace py_as, arr, lengths, float final_reward, float step_penalty):
        cdef VocabIdSeq vocab = np2vocab(arr, lengths, len(arr))
        self.ptr = new Env(py_ws.ptr, py_as.ptr, vocab, final_reward, step_penalty)
        self.word_space = py_ws
        self.action_space = py_as
        tn_cls = type(self).tn_cls
        self.start = wrap_node(tn_cls, self.ptr.start)
        self.end = wrap_node(tn_cls, self.ptr.end)

    def __dealloc__(self):
        del self.ptr


cdef inline PyTreeNode wrap_node(cls, TreeNode *ptr):
    cdef PyTreeNode py_tn = cls.__new__(cls, from_ptr=True)
    py_tn.ptr = ptr
    return py_tn


@cython.boundscheck(False)
def parallel_select(PyTreeNode py_root,
                    PyEnv py_env,
                    int num_sims,
                    int num_threads,
                    int depth_limit,
                    float puct_c,
                    int game_count,
                    float virtual_loss):
    if py_root.done:
        raise ValueError('Root is already the terminal state.')

    cdef TreeNode *root = py_root.ptr
    cdef Env *env = py_env.ptr

    cdef int i, n_steps_left
    cdef action_t best_i, action_id
    cdef TreeNode *node,
    cdef TreeNode *next_node
    cdef Action action
    steps_left = np.zeros([num_sims], dtype='long')
    cdef long[::1] steps_left_view = steps_left
    cdef vector[TNptr] selected = vector[TNptr](num_sims)

    with nogil:
        for i in prange(num_sims, num_threads=num_threads):
            node = root
            n_steps_left = depth_limit
            while n_steps_left > 0 and not node.is_leaf():
                best_i = node.select(puct_c, game_count, virtual_loss)
                action_id = node.action_allowed.at(best_i)
                action = env.action_space.get_action(action_id)
                next_node = env.apply_action(node, action)
                n_steps_left = n_steps_left - 1
                node = next_node
                # Terminate if stop action is chosen or it has reached the end.
                if node == nullptr or node.done:
                    break

            if node != nullptr:
                env.action_space.set_action_allowed(node)
            selected[i] = node
            steps_left_view[i] = n_steps_left

    tn_cls = type(py_root)
    return [wrap_node(tn_cls, ptr) if ptr != nullptr else None for ptr in selected], steps_left


@cython.boundscheck(False)
cdef object c_parallel_get_sparse_action_masks(vector[anyTNptr] nodes, int num_threads):
    cdef size_t n = nodes.size()
    cdef int i, j, k
    cdef anyTNptr node

    # First pass to get the maximum number of actions.
    lengths = np.zeros([n], dtype='long')
    cdef long[::1] lengths_view = lengths
    with nogil:
        for i in prange(n, num_threads=num_threads):
            lengths_view[i] = nodes[i].action_allowed.size()
    cdef size_t m = max(lengths)

    # Fill in the results
    arr = np.zeros([n, m], dtype='long')
    num_actions = np.zeros([n], dtype='long')
    action_masks = np.ones([n, m], dtype='bool')
    cdef long[:, ::1] arr_view = arr
    cdef long[::1] na_view = num_actions
    cdef bool[:, ::1] am_view = action_masks
    cdef vector[action_t] action_allowed
    with nogil:
        for i in prange(n, num_threads=num_threads):
            action_allowed = nodes[i].action_allowed
            k = action_allowed.size()
            for j in range(k):
                arr_view[i, j] = action_allowed[j]
            na_view[i] = k
            for j in range(k, m):
                am_view[i, j] = False
    return arr, action_masks, num_actions


# Use this to circumvent the issue of not being able to specifier a type identifier like vector[PyTreeNode].
cdef inline TreeNode *get_ptr(PyTreeNode py_node):
    return py_node.ptr
cdef inline DetachedTreeNode *get_dptr(PyDetachedTreeNode py_dnode):
    return py_dnode.ptr


@cython.boundscheck(False)
def parallel_get_sparse_action_masks(py_nodes, int num_threads):
    # Prepare the vector of (detached) tree nodes first.
    cdef size_t n = len(py_nodes)
    cdef vector[TNptr] nodes = vector[TNptr](n)
    cdef vector[DTNptr] dnodes = vector[DTNptr](n)
    is_detached = isinstance(py_nodes[0], PyDetachedTreeNode)
    for i, node in enumerate(py_nodes):
        if is_detached:
            dnodes[i] = get_dptr(node)
        else:
            nodes[i] = get_ptr(node)

    # Call the c function with the proper vector.
    if is_detached:
        return c_parallel_get_sparse_action_masks(dnodes, num_threads)
    else:
        return c_parallel_get_sparse_action_masks(nodes, num_threads)
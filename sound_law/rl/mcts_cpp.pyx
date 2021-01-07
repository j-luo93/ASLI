# distutils: language = c++
from .mcts_cpp cimport Mcts, Env, ActionSpace, SiteSpace, WordSpace, np2nested, TNptr, TreeNode, DetachedTreeNode, anyTNptr, uai_t, abc_t, combine, np2vector
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython.operator cimport dereference as deref, preincrement as inc

cimport cython

from sound_law.data.alphabet import PAD_ID

cdef extern from "limits.h":
    cdef uai_t ULONG_MAX

cdef abc_t NULL_ABC = (1 << 10) - 1
cdef uai_t NULL_ACTION = ULONG_MAX
cdef uai_t STOP = combine(NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC, NULL_ABC)

PyNull_abc = NULL_ABC
PyNull_action = NULL_ACTION
PyStop = STOP


cdef class PySiteSpace:
    cdef SiteSpace *ptr

    def __cinit__(self, abc_t sot_id, abc_t eot_id, abc_t any_id, abc_t emp_id):
        self.ptr = new SiteSpace(sot_id, eot_id, any_id, emp_id)

    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.size()

cdef class PyWordSpace:
    cdef WordSpace *ptr

    cdef public PySiteSpace site_space

    def __cinit__(self, PySiteSpace py_ss, float[:, ::1] np_dist_mat, float ins_cost):
        # Convert numpy dist_mat to vector.
        cdef size_t n = np_dist_mat.shape[0]
        cdef size_t m = np_dist_mat.shape[1]
        lengths = np.zeros(n, dtype='long')
        lengths.fill(m)
        cdef long[::1] lengths_view = lengths
        cdef vector[vector[float]] dist_mat = np2nested(np_dist_mat, lengths_view)
        self.ptr = new WordSpace(py_ss.ptr, dist_mat, ins_cost)
        self.site_space = py_ss

    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.size()

cdef class PyAction:
    cdef public uai_t action_id
    cdef public abc_t before_id
    cdef public abc_t after_id
    cdef public abc_t pre_id
    cdef public abc_t d_pre_id
    cdef public abc_t post_id
    cdef public abc_t d_post_id
    cdef public vector[abc_t] pre_cond
    cdef public vector[abc_t] post_cond

    def __cinit__(self,
                  abc_t before_id,
                  abc_t after_id,
                  abc_t pre_id,
                  abc_t d_pre_id,
                  abc_t post_id,
                  abc_t d_post_id,
                  action_id = None):
        self.before_id = before_id
        self.after_id = after_id
        self.pre_id = pre_id
        self.d_pre_id = d_pre_id
        self.post_id = post_id
        self.d_post_id = d_post_id
        self.pre_cond = vector[abc_t]()
        self.post_cond = vector[abc_t]()
        if d_pre_id != NULL_ABC:
            self.pre_cond.push_back(d_pre_id)
        if pre_id != NULL_ABC:
            self.pre_cond.push_back(pre_id)
        if post_id != NULL_ABC:
            self.post_cond.push_back(post_id)
        if d_post_id != NULL_ABC:
            self.post_cond.push_back(d_post_id)
        if action_id is None:
            action_id = combine(pre_id, d_pre_id, post_id, d_post_id, before_id, after_id)
        self.action_id = action_id

cdef class PyActionSpace:
    cdef ActionSpace *ptr

    def __cinit__(self, PySiteSpace py_ss, PyWordSpace py_ws, float dist_threshold, int site_threshold, *args, **kwargs):
        self.ptr = new ActionSpace(py_ss.ptr, py_ws.ptr, dist_threshold, site_threshold)

    def __dealloc__(self):
        del self.ptr

    def register_edge(self, abc_t before_id, abc_t after_id):
        self.ptr.register_edge(before_id, after_id)

    def get_action(self, action_id):
        return self.action_cls(get_before_id(action_id), get_after_id(action_id),
                               get_pre_id(action_id), get_d_pre_id(action_id),
                               get_post_id(action_id), get_d_post_id(action_id), action_id)

    def set_action_allowed(self, PyTreeNode tnode):
        self.ptr.set_action_allowed(tnode.ptr)

    def apply_action(self, id_seq, PyAction action):
        cdef abc_t[::1] np_id_seq = np.asarray(id_seq, dtype="ushort")
        cdef vector[abc_t] id_vec = np2vector(np_id_seq);
        cdef vector[abc_t] new_id_seq = self.ptr.apply_action(id_vec, action.action_id)
        return np.asarray(new_id_seq, dtype='ushort')

cdef class PyTreeNode:
    cdef TreeNode *ptr

    def __cinit__(self, from_ptr=False):
        if not from_ptr:
            raise TypeError(f'Cannot create a PyTreeNode object unless directly wrapping around a TreeNode pointer.')

    def __dealloc__(self):
        # Set ptr to NULL instead of destroying it.
        self.ptr = NULL

    def get_num_descendants(self):
        return self.ptr.get_num_descendants()

    def clear_stats(self, recursive: bool = False):
        self.ptr.clear_stats(recursive)

    @property
    def stopped(self) -> bool:
        return self.ptr.stopped

    @property
    def done(self) -> bool:
        return self.ptr.done

    @property
    def vocab_array(self):
        # Find the longest sequence.
        cdef int n = self.ptr.size()
        cdef int m = max([self.ptr.get_id_seq(i).size() for i in range(n)])
        # Fill in the array.
        arr = np.full([n, m], PAD_ID, dtype='long')
        cdef IdSeq id_seq
        for i in range(n):
            id_seq = self.ptr.get_id_seq(i)
            for j in range(id_seq.size()):
                arr[i, j] = id_seq[j]
        return arr

    @property
    def vocab(self):
        cdef int n = self.ptr.size()
        cdef VocabIdSeq vocab = VocabIdSeq(n)
        for i in range(n):
            vocab[i] = self.ptr.get_id_seq(i)
        return vocab

    @property
    def dist(self) -> float:
        return self.ptr.dist

    def is_leaf(self) -> bool:
        return self.ptr.is_leaf()

    def expand(self, float[::1] prior):
        cdef vector[float] prior_vec = np2vector(prior)
        self.ptr.expand(prior_vec)

    def add_noise(self, float[::1] noise, float noise_ratio):
        cdef vector[float] noise_vec = np2vector(noise)
        self.ptr.add_noise(noise_vec, noise_ratio)

    @property
    def num_actions(self):
        return self.ptr.action_allowed.size()

    @property
    def action_allowed(self):
        return np.asarray(self.ptr.action_allowed, dtype='long')

    @property
    def action_count(self):
        return np.asarray(self.ptr.action_count, dtype='long')

    @property
    def max_index(self):
        return self.ptr.max_index

    @property
    def max_action_id(self):
        return self.ptr.max_action_id

    @property
    def total_value(self):
        return np.asarray(self.ptr.total_value, dtype='float32')

    @property
    def prev_action(self):
        return self.ptr.prev_action

    def detach(self):
        cdef DetachedTreeNode *ptr = new DetachedTreeNode(self.ptr)
        cdef PyDetachedTreeNode py_dtn = PyDetachedTreeNode.__new__(PyDetachedTreeNode, from_ptr=True)
        py_dtn.ptr = ptr
        return py_dtn

cdef class PyDetachedTreeNode:
    cdef DetachedTreeNode *ptr

    def __cinit__(self, *, from_ptr=False):
        if not from_ptr:
            raise TypeError(f'Cannot create a PyDetachedTreeNode object unless directly wrapping around a DetachedTreeNode pointer.')

    def __dealloc__(self):
        del self.ptr

cdef inline PyTreeNode wrap_node(cls, TreeNode *ptr):
    cdef PyTreeNode py_tn = cls.__new__(cls, from_ptr=True)
    py_tn.ptr = ptr
    return py_tn

cdef class PyEnv:
    cdef Env *ptr

    cdef public PyWordSpace word_space
    cdef public PyActionSpace action_space
    cdef public PyTreeNode start
    cdef public PyTreeNode end

    tnode_cls = PyTreeNode

    def __cinit__(self, PyActionSpace py_as, PyWordSpace py_ws,
                  abc_t[:, ::1] np_start_ids, long[::1] start_lengths,
                  abc_t[:, ::1] np_end_ids, long[::1] end_lengths,
                  float final_reward, float step_penalty):
        cdef vector[vector[abc_t]] start_ids = np2nested(np_start_ids, start_lengths)
        cdef vector[vector[abc_t]] end_ids = np2nested(np_end_ids, end_lengths)
        self.ptr = new Env(py_as.ptr, py_ws.ptr, start_ids, end_ids, final_reward, step_penalty)
        tnode_cls = type(self).tnode_cls
        self.word_space = py_ws
        self.action_space = py_as
        self.start = wrap_node(tnode_cls, self.ptr.start)
        self.end = wrap_node(tnode_cls, self.ptr.end)

    def __dealloc__(self):
        del self.ptr

    def step(self, PyTreeNode node, int best_i, uai_t action_id):
        new_node = self.ptr.apply_action(node.ptr, best_i, action_id)
        tnode_cls = type(node)
        return wrap_node(tnode_cls, new_node)

cdef class PyMcts:
    cdef Mcts *ptr

    def __cinit__(self, PyEnv py_env, float puct_c, int game_count, float virtual_loss, int num_threads, *args, **kwargs):
        self.ptr = new Mcts(py_env.ptr, puct_c, game_count, virtual_loss, num_threads)

    def __dealloc__(self):
        del self.ptr

    def select(self, PyTreeNode py_tnode, int num_sims, int depth_limit):
        cdef vector[TNptr] selected = self.ptr.select(py_tnode.ptr, num_sims, depth_limit)
        cdef vector[int] steps_left_vec = vector[int](selected.size())
        cdef size_t i
        for i in range(selected.size()):
            steps_left_vec[i] = selected[i].depth
        steps_left = np.asarray(steps_left_vec, dtype='long')
        tnode_cls = type(py_tnode)
        return [wrap_node(tnode_cls, node) for node in selected], steps_left

    def backup(self, states, values):
        cdef size_t n = len(states)
        cdef vector[TNptr] states_vec = vector[TNptr](n)
        for i in range(n):
            states_vec[i] = get_ptr(states[i])
        cdef float[::1] np_values = np.asarray(values, dtype='float32')
        cdef vector[float] values_vec = np2vector(np_values)
        self.ptr.backup(states_vec, values_vec)

    def play(self, PyTreeNode py_tnode):
        cdef uai_t action_id = self.ptr.play(py_tnode.ptr)
        cdef float reward = py_tnode.ptr.rewards.at(action_id)
        cdef TreeNode *new_node = py_tnode.ptr.neighbors.at(action_id)
        tnode_cls = type(py_tnode)
        return action_id, wrap_node(tnode_cls, new_node), reward

    def enable_timer(self):
        self.ptr.env.action_space.timer.enable()

    def disable_timer(self):
        self.ptr.env.action_space.timer.disable()

    def show_timer_stats(self):
        self.ptr.env.action_space.timer.show_stats()

    def set_logging_options(self, int verbose_level, bool log_to_file):
        self.ptr.set_logging_options(verbose_level, log_to_file)


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
    cdef vector[uai_t] action_allowed
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


@cython.boundscheck(False)
cdef object c_parallel_stack_ids(vector[anyTNptr] nodes, int num_threads):
    cdef int i, j, k
    cdef size_t n = nodes.size()
    cdef size_t nw = deref(nodes.at(0)).size()

    # Get the max length first.
    lengths = np.zeros([n, nw], dtype='long')
    cdef long[:, ::1] lengths_view = lengths
    with nogil:
        for i in prange(n, num_threads=num_threads):
            for j in range(nw):
                lengths_view[i, j] = nodes[i].get_id_seq(j).size()
    cdef size_t m = lengths.max()

    arr = np.full([n, m, nw], PAD_ID, dtype='long')
    cdef long[:, :, ::1] arr_view = arr
    cdef IdSeq id_seq
    with nogil:
        for i in prange(n, num_threads=num_threads):
            for k in range(nw):
                id_seq = nodes[i].get_id_seq(k)
                for j in range(id_seq.size()):
                    arr_view[i, j, k] = id_seq[j]
    return arr


@cython.boundscheck(False)
def parallel_stack_ids(py_nodes, int num_threads):
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
        return c_parallel_stack_ids(dnodes, num_threads)
    else:
        return c_parallel_stack_ids(nodes, num_threads)


@cython.boundscheck(False)
def parallel_stack_policies(edges, int num_actions, int num_threads):
    cdef int n = len(edges)
    cdef vector[float[::1]] mcts_pi_vec = vector[float[::1]](n)
    cdef vector[size_t] pi_len = vector[size_t](n)
    cdef int i = 0
    for i in range(n):
        mcts_pi_vec[i] = edges[i].mcts_pi
        pi_len[i] = len(edges[i].mcts_pi)

    ret = np.zeros([n, num_actions], dtype='float32')
    cdef float[:, ::1] ret_view = ret
    with nogil:
        for i in prange(n, num_threads=num_threads):
            ret_view[i, :pi_len[i]] = mcts_pi_vec[i]
    return ret
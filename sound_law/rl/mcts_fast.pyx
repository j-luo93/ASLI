# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.list cimport list as cpp_list
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref
from cython.parallel import prange
from cython cimport view
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

cdef extern from "ActionSpace.cpp":
    pass

cdef extern from "Env.cpp":
    pass

cdef extern from "Word.cpp":
    pass

cdef extern from "Site.cpp":
    pass

cdef extern from "common.h":
    ctypedef unsigned short abc_t
    ctypedef float cost_t
    ctypedef float dist_t
    ctypedef short visit_t
    ctypedef unsigned int action_t
    ctypedef unsigned long node_t
    ctypedef vector[abc_t] IdSeq
    ctypedef vector[IdSeq] VocabIdSeq

cdef extern from "limits.h":
    cdef abc_t USHRT_MAX

cdef abc_t NULL_abc = USHRT_MAX

PyNull_abc = NULL_abc

# These are used by numpy's python api.
np_action_t = np.uint32
np_cost_t = np.float32

cdef extern from "TreeNode.h":
    cdef cppclass TreeNode nogil:
        ctypedef TreeNode * TNptr
        ctypedef pair[TNptr, float] Edge

        @staticmethod
        void set_dist_mat(vector[vector[cost_t]])
        @staticmethod
        void set_max_mode(bool)

        TreeNode(VocabIdSeq) except +
        TreeNode(VocabIdSeq, TreeNode *) except +

        bool has_acted(action_t)
        size_t size()
        void lock()
        void unlock()
        bool is_leaf()
        action_t get_best_i(float)
        void expand(vector[float])
        void virtual_backup(action_t, int, float)
        void backup(float, float, int, float)
        void reset()
        void play()
        cpp_list[pair[action_t, float]] get_path()
        vector[float] get_scores(float)
        void clear_subtree()
        void add_noise(vector[float], float)
        size_t get_num_allowed()

        TreeNode *parent_node
        VocabIdSeq vocab_i
        dist_t dist_to_end
        pair[action_t, action_t] prev_action
        vector[float] prior
        vector[action_t] action_allowed
        vector[visit_t] action_count
        visit_t visit_count
        vector[float] total_value
        unordered_map[action_t, Edge] edges
        vector[float] max_value
        bool done
        bool played
        node_t idx
    cdef cppclass DetachedTreeNode nogil:
        DetachedTreeNode(TreeNode *)

        VocabIdSeq vocab_i
        dist_t dist_to_end
        pair[action_t, action_t] prev_action
        vector[action_t] action_allowed
        bool done

        size_t size()

cdef extern from "Action.h":
    cdef cppclass Action nogil:
        action_t action_id
        abc_t before_id
        abc_t after_id
        vector[abc_t] pre_cond
        vector[abc_t] post_cond

        abc_t get_pre_id()
        abc_t get_post_id()
        abc_t get_d_pre_id()
        abc_t get_d_post_id()

ctypedef Action * Aptr

cdef extern from "ActionSpace.h":
    cdef cppclass ActionSpace nogil:
        @staticmethod
        void set_conditional(bool)

        ActionSpace()

        vector[Aptr] actions

        # void register_action(abc_t, abc_t, vector[abc_t], vector[abc_t])
        void register_edge(abc_t, abc_t)

        Action *get_action(action_t)
        void set_action_allowed(TreeNode *)
        size_t size()
        void clear_cache()
        size_t get_cache_size()
        vector[abc_t] expand_a2i()

ctypedef TreeNode * TNptr
ctypedef DetachedTreeNode * DTNptr

cdef extern from "Env.h":
    ctypedef pair[TNptr, float] Edge
    cdef cppclass Env nogil:
        Env(TreeNode *, TreeNode *, ActionSpace *, float, float) except +

        Edge step(TreeNode *, action_t, Action *) except +

        TreeNode *init_node
        TreeNode *end_node

cdef extern from "Word.h":
    cdef cppclass Word nogil:
        @staticmethod
        void set_end_words(VocabIdSeq)

cdef inline VocabIdSeq np2vocab(long[:, ::1] arr,
                                long[::1] lengths,
                                size_t n) except *:
    cdef size_t i, j, m
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
    cdef size_t n = vocab_i.size()
    cdef size_t m = 0
    # Find the longest sequence.
    cdef size_t i, j
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
    action_t

cdef inline vector[convertible] np2vector(convertible[::1] arr, size_t n) except *:
    cdef size_t i
    cdef vector[convertible] vec = vector[convertible](n)
    for i in range(n):
        vec[i] = arr[i]
    return vec

cdef inline object get_py_edge(PyTreeNode node, Edge edge):
    cdef TreeNode * next_node = edge.first
    cdef bool done = node.ptr.done
    cdef float reward = edge.second
    return wrap_node(type(node), next_node), done, reward


cdef class PyTreeNode:
    cdef TNptr ptr
    cdef public PyTreeNode end_node

    def __dealloc__(self):
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
        cdef size_t n = arr.shape[0]
        assert n == lengths.shape[0], '`arr` and `lengths` must have the same length.'
        cdef long[::1] lengths_view = lengths

        cdef VocabIdSeq vocab_i = np2vocab(arr_view, lengths_view, n)
        if end_node is None:
            self.ptr = new TreeNode(vocab_i)
        else:
            self.ptr = new TreeNode(vocab_i, end_node.ptr)
        self.end_node = end_node

    def __len__(self):
        return self.ptr.size()

    def get_num_allowed(self):
        return self.ptr.get_num_allowed()

    def get_scores(self, float puct_c):
        return np.asarray(self.ptr.get_scores(puct_c), dtype='float32')

    @property
    def idx(self):
        return self.ptr.idx

    @property
    def vocab_array(self):
        return np.asarray(vocab2np(self.ptr.vocab_i), dtype='long')

    @property
    def vocab(self):
        return self.ptr.vocab_i

    @property
    def parent(self):
        if self.ptr.parent_node == nullptr:
            return None
        return wrap_node(type(self), self.ptr.parent_node)

    @property
    def prior(self):
        return np.asarray(self.ptr.prior, dtpye='float32')

    @property
    def visit_count(self):
        return self.ptr.visit_count

    @property
    def action_count(self):
        return np.asarray(self.ptr.action_count, dtype='long')

    @property
    def total_value(self):
        return np.asarray(self.ptr.total_value, dtype='float32')

    @property
    def max_value(self):
        return np.asarray(self.ptr.max_value, dtype='float32')

    @property
    def played(self):
        return self.ptr.played

    def get_path(self):
        return self.ptr.get_path()

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

    def is_leaf(self):
        return self.ptr.is_leaf()

    def expand(self, float[::1] prior):
        cdef size_t n = len(prior)
        cdef vector[float] prior_vec = np2vector(prior, n)
        self.ptr.expand(prior_vec)

    def backup(self, float value, float mixing, int game_count, float virtual_loss):
        self.ptr.backup(value, mixing, game_count, virtual_loss)

    def reset(self):
        self.ptr.reset()

    def play(self):
        self.ptr.play()

    def __eq__(self, PyTreeNode other):
        return self.ptr.vocab_i == other.ptr.vocab_i

    @property
    def prev_action(self):
        return self.ptr.prev_action

    @property
    def dist_to_end(self):
        return self.ptr.dist_to_end

    @property
    def done(self):
        return self.ptr.done

    @property
    def action_allowed(self):
        return np.asarray(self.ptr.action_allowed, dtype='long')

    def get_edge(self, action_t action_id):
        cdef Edge edge
        if self.ptr.has_acted(action_id):
            edge = self.ptr.edges.at(action_id)
            return get_py_edge(self, edge)
        raise ValueError(f'Action {action_id} has not been explored.')

    def clear_subtree(self):
        self.ptr.clear_subtree()

    def add_noise(self, float [::1] noise, float noise_ratio):
        cdef size_t n = len(noise)
        cdef vector[float] noise_vec = np2vector(noise, n)
        self.ptr.add_noise(noise_vec, noise_ratio)

    @staticmethod
    def set_dist_mat(object np_dist_mat):
        cdef cost_t[:, ::1] dist_view = np_dist_mat.astype(np_cost_t)
        cdef size_t n = np_dist_mat.shape[0]
        cdef size_t m = np_dist_mat.shape[1]
        cdef vector[vector[cost_t]] dist_mat = vector[vector[cost_t]](n)
        cdef size_t i, j
        cdef vector[cost_t] vec
        for i in range(n):
            vec = vector[cost_t](m)
            for j in range(m):
                vec[j] = dist_view[i, j]
            dist_mat[i] = vec
        TreeNode.set_dist_mat(dist_mat)

    @staticmethod
    def set_max_mode(bool max_mode):
        TreeNode.set_max_mode(max_mode)

    def detach(self):
        cdef DetachedTreeNode *ptr = new DetachedTreeNode(self.ptr)
        cdef PyDetachedTreeNode py_dtn = PyDetachedTreeNode.__new__(PyDetachedTreeNode)
        py_dtn.ptr = ptr
        return py_dtn


cdef class PyDetachedTreeNode:
    cdef DetachedTreeNode *ptr

    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.size()

    @property
    def vocab_array(self):
        return np.asarray(vocab2np(self.ptr.vocab_i), dtype='long')

    @property
    def vocab(self):
        return self.ptr.vocab_i

    @property
    def prev_action(self):
        return self.ptr.prev_action

    @property
    def dist_to_end(self):
        return self.ptr.dist_to_end

    @property
    def done(self):
        return self.ptr.done

    @property
    def action_allowed(self):
        return np.asarray(self.ptr.action_allowed, dtype='long')


cdef class PyAction:
    """This is a wrapper class for c++ class Action. It should be created by a PyActionSpace object with registered actions."""
    cdef Aptr ptr

    def __cinit__(self, *args, bool from_ptr=False, **kwargs):
        assert from_ptr, 'You should only create this object by calling `from_ptr`.'

    def __dealloc__(self):
        self.ptr = NULL
        # del self.ptr

    @property
    def action_id(self):
        return self.ptr.action_id

    @property
    def before_id(self):
        return self.ptr.before_id

    @property
    def after_id(self):
        return self.ptr.after_id

    @property
    def pre_id(self):
        if self.ptr.pre_cond.size() > 0:
            return self.ptr.pre_cond.back()
        return -1

    @property
    def d_pre_id(self):
        if self.ptr.pre_cond.size() > 1:
            return self.ptr.pre_cond.front()
        return -1

    @property
    def post_id(self):
        if self.ptr.post_cond.size() > 0:
            return self.ptr.post_cond.front()
        return -1

    @property
    def d_post_id(self):
        if self.ptr.post_cond.size() > 1:
            return self.ptr.post_cond.back()
        return -1

    @property
    def pre_cond(self):
        return self.ptr.pre_cond

    @property
    def post_cond(self):
        return self.ptr.post_cond


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

    # def register_action(self, abc_t before_id, abc_t after_id, pre_cond=None, post_cond=None):
    #     pre_cond = pre_cond or list()
    #     post_cond = post_cond or list()
    #     cdef vector[abc_t] cpp_pre_cond = vector[abc_t](len(pre_cond))
    #     cdef vector[abc_t] cpp_post_cond = vector[abc_t](len(post_cond))
    #     cdef size_t i
    #     for i in range(len(pre_cond)):
    #         cpp_pre_cond[i] = pre_cond[i]
    #     for i in range(len(post_cond)):
    #         cpp_post_cond[i] = post_cond[i]
    #     self.ptr.register_action(before_id, after_id, cpp_pre_cond, cpp_post_cond)

    def register_edge(self, abc_t before_id, abc_t after_id):
        self.ptr.register_edge(before_id, after_id)

    @staticmethod
    def set_conditional(bool conditional):
        ActionSpace.set_conditional(conditional)

    def set_action_allowed(self, PyTreeNode node):
        self.ptr.set_action_allowed(node.ptr)

    def get_action_mask(self, PyTreeNode node):
        cdef vector[action_t] action_allowed = node.action_allowed
        ret = np.zeros([len(self)], dtype='bool')
        for i in range(action_allowed.size()):
            ret[action_allowed[i]] = True
        return  ret

    def get_action(self, action_t action_id):
        if action_id >= len(self) or action_id < 0:
            raise ValueError(f'Action id out of bound.')
        cdef Action *action = self.ptr.get_action(action_id)
        action_cls = type(self).action_cls
        return wrap_action(action_cls, action)

    def gather(self, attr, int num_workers):
        assert attr in ['before_id', 'after_id', 'pre_id', 'post_id', 'd_pre_id', 'd_post_id']
        cdef size_t i
        cdef size_t n = self.ptr.size()
        ret = np.zeros([n], dtype='long')
        cdef long[::1] ret_view = ret
        cdef Action *action
        cdef abc_t idx
        with nogil:
            for i in prange(n, num_threads=num_workers):
                action = self.ptr.get_action(i)
                if attr == 'before_id':
                    idx = action.before_id
                elif attr == 'after_id':
                    idx = action.after_id
                elif attr == 'pre_id':
                    idx = action.get_pre_id()
                elif attr == 'd_pre_id':
                    idx = action.get_d_pre_id()
                elif attr == 'post_id':
                    idx = action.get_post_id()
                else:
                    idx = action.get_d_post_id()
                ret_view[i] = idx
        ret[ret == NULL_abc] = -1
        return ret

    def __len__(self):
        return self.ptr.size()

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_action(i)

    def clear_cache(self):
        self.ptr.clear_cache()

    @property
    def cache_size(self):
        return self.ptr.get_cache_size()

    def expand_a2i(self):
        return np.asarray(self.ptr.expand_a2i(), dtype='long')

ctypedef Env * Envptr

cdef class PyEnv:
    cdef Envptr ptr

    def __cinit__(self,
                  PyTreeNode init_node,
                  PyTreeNode end_node,
                  PyActionSpace py_as,
                  float final_reward,
                  float step_penalty,
                  *args, **kwargs):
        self.ptr = new Env(init_node.ptr, end_node.ptr, py_as.ptr, final_reward, step_penalty)

    def __dealloc__(self):
        self.ptr = NULL

    def step(self, PyTreeNode node, action_t best_i, PyAction action):
        cdef Edge edge = self.ptr.step(node.ptr, best_i, action.ptr)
        return get_py_edge(node, edge)

# IDEA(j_luo) rename node to state?
cpdef object parallel_select(PyTreeNode py_root,
                             PyTreeNode py_end,
                             PyActionSpace py_as,
                             PyEnv py_env,
                             int num_sims,
                             int num_threads,
                             int depth_limit,
                             float puct_c,
                             int game_count,
                             float virtual_loss):
    cdef TreeNode *end = py_end.ptr
    cdef TreeNode *root = py_root.ptr
    cdef Env *env = py_env.ptr

    cdef TreeNode *node
    cdef float reward
    cdef long n_steps_left
    cdef size_t i
    cdef action_t action_id, best_i
    cdef vector[action_t] action_allowed
    cdef Edge edge
    cdef Action *action
    cdef vector[TNptr] selected = vector[TNptr](num_sims)
    cdef ActionSpace *action_space = py_as.ptr
    steps_left = np.zeros([num_sims], dtype='long')
    cdef long[::1] steps_left_view = steps_left

    with nogil:
        for i in prange(num_sims, num_threads=num_threads):
            node = root
            n_steps_left = depth_limit
            while n_steps_left > 0 and not node.is_leaf():
                node.lock()
                best_i = node.get_best_i(puct_c)
                action_allowed = node.action_allowed
                action_id = action_allowed[best_i]
                action = action_space.get_action(action_id)
                edge = env.step(node, best_i, action)
                node.virtual_backup(best_i, game_count, virtual_loss)
                node.unlock()

                n_steps_left = n_steps_left - 1
                if node.done:
                    break

                node = edge.first
            selected[i] = node
            steps_left_view[i] = n_steps_left
        for i in prange(num_sims, num_threads=num_threads):
            node = selected[i]
            # This is in-place.
            node.lock()
            action_space.set_action_allowed(node)
            node.unlock()
    tn_cls = type(py_root)
    return [wrap_node(tn_cls, ptr) for ptr in selected], steps_left

# Use this to circumvent the issue of not being able to specifier a type identifier like vector[PyTreeNode].
cdef inline TreeNode *get_ptr(PyTreeNode py_node):
    return py_node.ptr

cdef inline DetachedTreeNode *get_dptr(PyDetachedTreeNode py_node):
    return py_node.ptr

cpdef object parallel_get_action_masks(object py_nodes, PyActionSpace py_as, int num_threads):
    cdef size_t n = len(py_nodes)
    cdef size_t m = len(py_as)
    cdef size_t i, j
    cdef TreeNode *node
    cdef vector[TNptr] nodes = vector[TNptr](n)
    cdef vector[action_t] action_allowed
    for i in range(n):
        nodes[i] = get_ptr(py_nodes[i])

    arr = np.zeros([n, m], dtype='bool')
    cdef bool[:, ::1] arr_view = arr
    with nogil:
        for i in prange(n, num_threads=num_threads):
            node = nodes[i]
            action_allowed = node.action_allowed
            m = action_allowed.size()
            for j in range(m):
                arr_view[i, action_allowed[j]] = True
    return arr

cpdef object parallel_get_sparse_action_masks(object py_nodes, int num_threads):
    cdef size_t n = len(py_nodes)
    cdef size_t i, j, k
    cdef DetachedTreeNode *dnode
    cdef TreeNode *node
    cdef vector[DTNptr] dnodes = vector[DTNptr](n)
    cdef vector[TNptr] nodes = vector[TNptr](n)
    cdef vector[action_t] action_allowed
    # HACK(j_luo) very hacky here
    cdef bool is_detached = isinstance(py_nodes[0], PyDetachedTreeNode)
    for i in range(n):
        if is_detached:
            dnodes[i] = get_dptr(py_nodes[i])
        else:
            nodes[i] = get_ptr(py_nodes[i])

    # First pass to get the maximum number of actions.
    lengths = np.zeros([n], dtype='long')
    cdef long[::1] lengths_view = lengths
    with nogil:
        for i in prange(n, num_threads=num_threads):
            if is_detached:
                action_allowed = dnodes[i].action_allowed
            else:
                action_allowed = nodes[i].action_allowed
            lengths_view[i] = action_allowed.size()
    cdef size_t m = max(lengths)

    arr = np.zeros([n, m], dtype='long')
    num_actions = np.zeros([n], dtype='long')
    action_masks = np.ones([n, m], dtype='bool')
    cdef long[:, ::1] arr_view = arr
    cdef long[::1] na_view = num_actions
    cdef bool[:, ::1] am_view = action_masks
    with nogil:
        for i in prange(n, num_threads=num_threads):
            if is_detached:
                action_allowed = dnodes[i].action_allowed
            else:
                action_allowed = nodes[i].action_allowed
            k = action_allowed.size()
            for j in range(k):
                arr_view[i, j] = action_allowed[j]
            na_view[i] = k
            for j in range(k, m):
                am_view[i, j] = False
    return arr, action_masks, num_actions

cpdef object parallel_stack_ids(object py_nodes, int num_threads):
    cdef size_t n = len(py_nodes)
    # HACK(j_luo) very hacky here
    cdef vector[TNptr] nodes = vector[TNptr](n)
    cdef vector[DTNptr] dnodes = vector[DTNptr](n)
    cdef size_t i, j, k, m
    cdef bool is_detached = isinstance(py_nodes[0], PyDetachedTreeNode)
    for i in range(n):
        if is_detached:
            dnodes[i] = get_dptr(py_nodes[i])
        else:
            nodes[i] = get_ptr(py_nodes[i])
    cdef size_t nw = len(py_nodes[0])

    lengths = np.zeros([n, nw], dtype='long')
    cdef long[:, ::1] lengths_view = lengths
    cdef VocabIdSeq vocab_i
    with nogil:
        for i in prange(n, num_threads=num_threads):
            if is_detached:
                vocab_i = dnodes[i].vocab_i
            else:
                vocab_i = nodes[i].vocab_i
            for j in range(nw):
                lengths_view[i, j] = vocab_i[j].size()

    m = lengths.max()
    arr = np.full([n, m, nw], PAD_ID, dtype='long')
    cdef long[:, :, ::1] arr_view = arr
    cdef IdSeq id_seq
    with nogil:
        for i in prange(n, num_threads=num_threads):
            if is_detached:
                vocab_i = dnodes[i].vocab_i
            else:
                vocab_i = nodes[i].vocab_i
            for k in range(nw):
                id_seq = vocab_i[k]
                for j in range(id_seq.size()):
                    arr_view[i, j, k] = id_seq[j]
    return  arr

cpdef object parallel_stack_policies(object edges, action_t num_actions, int num_threads):
    cdef long n = len(edges)
    cdef vector[float[::1]] mcts_pi_vec = vector[float[::1]](n)
    cdef vector[size_t] pi_len = vector[size_t](n)
    cdef size_t i = 0
    for i in range(n):
        mcts_pi_vec[i] = edges[i].mcts_pi
        pi_len[i] = len(edges[i].mcts_pi)

    ret = np.zeros([n, num_actions], dtype='float32')
    cdef float[:, ::1] ret_view = ret
    with nogil:
        for i in prange(n, num_threads=num_threads):
            ret_view[i, :pi_len[i]] = mcts_pi_vec[i]
    return ret

cpdef object parallel_gather_action_info(PyActionSpace py_as, action_ids, int num_threads):
    cdef size_t i, j, k
    cdef abc_t idx

    cdef size_t n = action_ids.shape[0]
    cdef size_t m = action_ids.shape[1]
    cdef long[:, ::1] id_view = action_ids
    ret = np.zeros([n, m * 6], dtype='long')
    cdef long[:, ::1] ret_view = ret
    cdef ActionSpace *action_space = py_as.ptr
    cdef Action *action
    with nogil:
        for i in prange(n, num_threads=num_threads):
            for j in range(m):
                k = 6 * j
                action = action_space.get_action(id_view[i, j])
                ret_view[i, k] = action.before_id
                ret_view[i, k + 1] = action.after_id
                ret_view[i, k + 2] = action.get_pre_id()
                ret_view[i, k + 3] = action.get_d_pre_id()
                ret_view[i, k + 4] = action.get_post_id()
                ret_view[i, k + 5] = action.get_d_post_id()
    return ret

cpdef set_end_words(PyTreeNode py_node):
    cdef TreeNode *node = py_node.ptr
    Word.set_end_words(node.vocab_i)
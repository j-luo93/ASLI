# distutils: language = c++
from .mcts_cpp cimport TreeNode, IdSeq, VocabIdSeq, Env, Mcts
from .mcts_cpp cimport Stress, NOSTRESS, STRESSED, UNSTRESSED
from .mcts_cpp cimport SpecialType, NONE, CLL, CLR, VS, GBJ, GBW
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython.operator cimport dereference as deref, preincrement as inc
cimport cython

import sound_law.data.alphabet as alphabet

PyNoStress = <int>NOSTRESS
PyStressed = <int>STRESSED
PyUnstressed = <int>UNSTRESSED

PyST_NONE = <abc_t>NONE
PyST_CLL = <abc_t>CLL
PyST_CLR = <abc_t>CLR
PyST_VS = <abc_t>VS
PyST_GBJ = <abc_t>GBJ
PyST_GBW = <abc_t>GBW

cdef class PyTreeNode:
    cdef TreeNode *ptr

    def __cinit__(self, from_ptr=False):
        if not from_ptr:
            raise TypeError(f'Cannot create a PyTreeNode object unless directly wrapping around a TreeNode pointer.')

    def __dealloc__(self):
        # Set ptr to NULL instead of destroying it.
        self.ptr = NULL

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
        arr = np.full([n, m], alphabet.PAD_ID, dtype='long')
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
    def num_actions(self) -> int:
        return self.ptr.get_num_actions()

    def add_noise(self, float[::1] noise, float noise_ratio):
        self.ptr.add_noise(np2vector(noise), noise_ratio)

    @property
    def dist(self) -> float:
        return self.ptr.dist

    def is_leaf(self) -> bool:
        return self.ptr.is_leaf()

    # def expand(self, float[::1] prior):
    #     cdef vector[float] prior_vec = np2vector(prior)
    #     self.ptr.expand(prior_vec)

    # def add_noise(self, float[::1] noise, float noise_ratio):
    #     cdef vector[float] noise_vec = np2vector(noise)
    #     self.ptr.add_noise(noise_vec, noise_ratio)

    # @property
    # def num_actions(self):
    #     return self.ptr.action_allowed.size()

    # @property
    # def action_allowed(self):
    #     return np.asarray(self.ptr.action_allowed, dtype='long')

    @property
    def chosen_char(self):
        return (<BaseNode *>self.ptr).chosen_char

    @property
    def parent(self):
        cdef BaseNode *node = <BaseNode *>self.ptr
        if node.parent == NULL:
            return None
        # We need to go back 7 times.
        for i in range(7):
            node = node.parent
        return wrap_node(type(self), <TreeNode *>node)

    @property
    def action_counts(self):
        return np.asarray((<BaseNode *>self.ptr).action_counts, dtype='long')

    # @property
    # def max_index(self):
    #     return self.ptr.max_index

    # @property
    # def max_action_id(self):
    #     return self.ptr.max_action_id

    @property
    def total_values(self):
        return np.asarray((<BaseNode *>self.ptr).total_values, dtype='float32')

    # @property
    # def prev_action(self):
    #     return self.ptr.prev_action

    # def detach(self):
    #     cdef DetachedTreeNode *ptr = new DetachedTreeNode(self.ptr)
    #     cdef PyDetachedTreeNode py_dtn = PyDetachedTreeNode.__new__(PyDetachedTreeNode, from_ptr=True)
    #     py_dtn.ptr = ptr
    #     return py_dtn

cdef inline PyTreeNode wrap_node(cls, TreeNode *ptr):
    cdef PyTreeNode py_tn = cls.__new__(cls, from_ptr=True)
    py_tn.ptr = ptr
    return py_tn

cdef class PyEnvOpt:
    cdef EnvOpt c_obj

    def __cinit__(self,
                  abc_t[:, ::1] np_start_ids, long[::1] start_lengths,
                  abc_t[:, ::1] np_end_ids, long[::1] end_lengths,
                  float final_reward, float step_penalty):
        cdef vector[vector[abc_t]] start_ids = np2nested(np_start_ids, start_lengths)
        cdef vector[vector[abc_t]] end_ids = np2nested(np_end_ids, end_lengths)
        self.c_obj = EnvOpt()
        self.c_obj.start_ids = start_ids
        self.c_obj.end_ids = end_ids
        self.c_obj.final_reward = final_reward
        self.c_obj.step_penalty = step_penalty


cdef class PyActionSpaceOpt:
    cdef ActionSpaceOpt c_obj

    def __cinit__(self, abc_t null_id, abc_t emp_id, abc_t sot_id, abc_t eot_id, abc_t any_id, abc_t any_s_id, abc_t any_uns_id, abc_t glide_j, abc_t glide_w):
        self.c_obj = ActionSpaceOpt()
        self.c_obj.null_id = null_id
        self.c_obj.emp_id = emp_id
        self.c_obj.sot_id = sot_id
        self.c_obj.eot_id = eot_id
        self.c_obj.any_id = any_id
        self.c_obj.any_s_id = any_s_id
        self.c_obj.any_uns_id = any_uns_id
        self.c_obj.glide_j = glide_j
        self.c_obj.glide_w = glide_w

cdef class PyWordSpaceOpt:
    cdef WordSpaceOpt c_obj

    def __cinit__(self, float[:, ::1] np_dist_mat, float ins_cost,
                  bool[::1] np_is_vowel, int[::1] np_unit_stress,
                  abc_t[::1] np_unit2base, abc_t[::1] np_unit2stressed,
                  abc_t[::1] np_unit2unstressed):
        cdef size_t n = np_dist_mat.shape[0]
        cdef size_t m = np_dist_mat.shape[1]
        cdef long[::1] lengths = np.zeros(n, dtype='long')
        for i in range(n):
            lengths[i] = m

        self.c_obj = WordSpaceOpt()
        self.c_obj.dist_mat = np2nested(np_dist_mat, lengths)
        self.c_obj.ins_cost = ins_cost
        self.c_obj.is_vowel = np2vector(np_is_vowel)
        cdef size_t num_abc = len(np_is_vowel)
        cdef vector[Stress] unit_stress = vector[Stress](num_abc)
        for i in range(num_abc):
            unit_stress[i] = <Stress>np_unit_stress[i]
        self.c_obj.unit_stress = unit_stress
        self.c_obj.unit2base = np2vector(np_unit2base)
        self.c_obj.unit2stressed = np2vector(np_unit2stressed)
        self.c_obj.unit2unstressed = np2vector(np_unit2unstressed)

cdef class PyMctsOpt:
    cdef MctsOpt c_obj

    def __cinit__(self,
                  float puct_c,
                  int game_count,
                  float virtual_loss,
                  int num_threads):
        self.c_obj = MctsOpt()
        self.c_obj.puct_c = puct_c
        self.c_obj.game_count = game_count
        self.c_obj.virtual_loss = virtual_loss
        self.c_obj.num_threads = num_threads

cdef class PyEnv:
    cdef Env *ptr

    cdef public PyTreeNode start
    cdef public PyTreeNode end

    tnode_cls = PyTreeNode

    def __cinit__(self, PyEnvOpt py_env_opt, PyActionSpaceOpt py_as_opt, PyWordSpaceOpt py_ws_opt, *args, **kwargs):
        self.ptr = new Env(py_env_opt.c_obj, py_as_opt.c_obj, py_ws_opt.c_obj)
        tnode_cls = type(self).tnode_cls
        self.start = wrap_node(tnode_cls, self.ptr.start)
        self.end = wrap_node(tnode_cls, self.ptr.end)

    def __dealloc__(self):
        del self.ptr

    def get_edit_dist(self, seq1, seq2):
        return self.ptr.get_edit_dist(seq1, seq2)

    def apply_action(self,
                     PyTreeNode py_node,
                     abc_t before_id,
                     abc_t after_id,
                     abc_t pre_id,
                     abc_t d_pre_id,
                     abc_t post_id,
                     abc_t d_post_id,
                     special_type: Optional[str] = None):
        cdef SpecialType st
        if special_type is None:
            st = NONE
        elif special_type == 'VS':
            st = VS
        elif special_type == 'CLL':
            st = CLL
        elif special_type == 'CLR':
            st = CLR
        elif special_type == 'GBJ':
            st = GBJ
        elif special_type == 'GBW':
            st = GBW

        cdef TreeNode *new_node = self.ptr.apply_action(py_node.ptr, before_id, after_id, pre_id, d_pre_id, post_id, d_post_id, st)
        tnode_cls = type(self).tnode_cls
        return wrap_node(tnode_cls, new_node)

    def register_permissible_change(self, abc_t unit1, abc_t unit2):
        self.ptr.register_permissible_change(unit1, unit2)

    def register_cl_map(self, abc_t unit1, abc_t unit2):
        self.ptr.register_cl_map(unit1, unit2)

    def register_gbj_map(self, abc_t unit1, abc_t unit2):
        self.ptr.register_gbj_map(unit1, unit2)

    def register_gbw_map(self, abc_t unit1, abc_t unit2):
        self.ptr.register_gbw_map(unit1, unit2)

    def clear_stats(self, PyTreeNode py_node, bool recursive):
        self.ptr.clear_stats(py_node.ptr, recursive)

    def clear_priors(self, PyTreeNode py_node, bool recursive):
        self.ptr.clear_priors(py_node.ptr, recursive)

    def prune(self, PyTreeNode py_node):
        self.ptr.prune(py_node.ptr)

    @property
    def num_words(self) -> int:
        return self.ptr.get_num_words()

    def evaluate(self, PyTreeNode py_node, float[:, ::1] np_meta_priors, float[::1] np_special_priors):
        cdef long[::1] lengths = np.full([6], np_meta_priors.shape[1], dtype='long')
        cdef vector[vector[float]] meta_priors = np2nested(np_meta_priors, lengths)
        cdef vector[float] special_priors = np2vector(np_special_priors)
        self.ptr.evaluate(py_node.ptr, meta_priors, special_priors)

    # def step(self, PyTreeNode node, int best_i, uai_t action_id):
    #     new_node = self.ptr.apply_action(node.ptr, best_i, action_id)
    #     tnode_cls = type(node)
    #     return wrap_node(tnode_cls, new_node)

cdef inline TreeNode *get_ptr(PyTreeNode py_node):
    return py_node.ptr

cdef class PyMcts:
    cdef Mcts *ptr

    cdef public PyEnv env

    def __cinit__(self, PyEnv py_env, PyMctsOpt py_mcts_opt, *args, **kwargs):
        self.ptr = new Mcts(py_env.ptr, py_mcts_opt.c_obj)
        self.env = py_env

    def __dealloc__(self):
        del self.ptr

    def select(self, PyTreeNode py_tnode, int num_sims, int depth_limit):
        cdef vector[TNptr] selected = self.ptr.select(py_tnode.ptr, num_sims, depth_limit)
        cdef vector[int] steps_left_vec = vector[int](selected.size())
        for i in range(selected.size()):
            steps_left_vec[i] = selected[i].depth
        steps_left = np.asarray(steps_left_vec, dtype='long')
        tnode_cls = type(py_tnode)
        return [wrap_node(tnode_cls, node) for node in selected], steps_left

    def backup(self, states, vector[float] values):
        cdef vector[TNptr] states_vec = vector[TNptr]()
        states_vec.reserve(len(states))
        for state in states:
            states_vec.push_back(get_ptr(state))
        self.ptr.backup(states_vec, values)

    def play(self, PyTreeNode py_tnode):
        return wrap_node(type(py_tnode), self.ptr.play(py_tnode.ptr))
        # cdef FullActionPath full_action = self.ptr.play(py_tnode.ptr)
        # return wrap_node(type(py_tnode), full_action.first.first), full_action.first.second, full_action.second

    # def enable_timer(self):
    #     stats.enable_timer()

    # def disable_timer(self):
    #     stats.disable_timer()

    # def show_stats(self):
    #     stats.show_stats()

    # def set_logging_options(self, int verbose_level, bool log_to_file):
    #     self.ptr.set_logging_options(verbose_level, log_to_file)

def parallel_stack_ids(py_nodes, int num_threads):
    cdef vector[TNptr] nodes = vector[TNptr]()
    for node in py_nodes:
        nodes.push_back(get_ptr(node))
    return c_parallel_stack_ids(nodes, num_threads)

@cython.boundscheck(False)
cdef object c_parallel_stack_ids(vector[TNptr] nodes, int num_threads):
    # Get the max length first.
    cdef int i, j, k
    cdef size_t m = 0
    cdef size_t n = nodes.size()
    cdef size_t nw = nodes[0].size()
    for i in range(n):
        for j in range(nw):
            m = max(m, nodes[i].get_id_seq(j).size())

    cdef long[:, :, ::1] arr = np.full([n, nw, m], alphabet.PAD_ID, dtype='long')
    cdef IdSeq id_seq
    with nogil:
        for i in prange(n, num_threads=num_threads):
            for j in range(nw):
                id_seq = nodes[i].get_id_seq(j)
                for k in range(id_seq.size()):
                    arr[i, j, k] = id_seq[k]
    return np.asarray(arr)

@cython.boundscheck(False)
cdef object c_parallel_stack_actions(vector[BNptr] nodes, int num_threads):
    cdef size_t i, j
    cdef size_t n = nodes.size()
    cdef size_t m = 0
    cdef vector[abc_t] pc
    cdef vector[visit_t] ac
    cdef float vc
    for i in range(n):
        m = max(m, nodes[i].permissible_chars.size())
    cdef long[:, ::1] actions = np.full([n, m], alphabet.SENTINEL_ID, dtype='long')
    cdef float[:, ::1] mcts_pis = np.zeros([n, m], dtype='float32')
    with nogil:
        for i in prange(n, num_threads=num_threads):
            pc = nodes[i].permissible_chars
            ac = nodes[i].action_counts
            vc = <float>(nodes[i].visit_count)
            for j in range(pc.size()):
                actions[i, j] = pc[j]
                mcts_pis[i, j] = <float>(ac[j]) / vc
    return np.asarray(actions), np.asarray(mcts_pis)

# @cython.boundscheck(False)
def parallel_gather_trajectory(PyTreeNode last_state, int num_threads):
    cdef BaseNode *node = <BaseNode *>last_state.ptr
    cdef list[BNptr] base_nodes_list = list[BNptr]()
    # Reverse the backtracking order to get the right order.
    while node != NULL:
        base_nodes_list.push_front(node)
        node = node.parent
    cdef vector[BNptr] base_nodes = vector[BNptr]()
    cdef list[BNptr].iterator it = base_nodes_list.begin()
    while it != base_nodes_list.end():
        base_nodes.push_back(deref(it))
        it = inc(it)

    cdef vector[TNptr] tree_nodes = vector[TNptr]()
    cdef vector[abc_t] actions = vector[abc_t]()
    cdef vector[float] rewards = vector[float]()
    cdef vector[float] qs = vector[float]()
    cdef int chosen_index
    for i in range(base_nodes.size()):
        node = base_nodes[i]
        if node.is_tree_node():
            tree_nodes.push_back(<TreeNode *>node)
        if node.parent != NULL:
            chosen_index = node.chosen_char.first
            actions.push_back(node.chosen_char.second)
            qs.push_back(node.parent.total_values[chosen_index] / node.parent.action_counts[chosen_index])
            if node.is_tree_node():
                rewards.push_back((<TransitionNode *>(node.parent)).rewards[chosen_index])
    id_seqs = c_parallel_stack_ids(tree_nodes, num_threads)
    # We don't need the last state's permissible actions since it's not explored.
    base_nodes.pop_back()
    permissible_actions, mcts_pis = c_parallel_stack_actions(base_nodes, num_threads)
    return id_seqs, np.asarray(actions), np.asarray(rewards), permissible_actions, mcts_pis, np.asarray(qs)
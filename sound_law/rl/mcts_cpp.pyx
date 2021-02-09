# distutils: language = c++
from .mcts_cpp cimport TreeNode, IdSeq, VocabIdSeq, Env, Mcts
from .mcts_cpp cimport Stress, NOSTRESS, STRESSED, UNSTRESSED
from .mcts_cpp cimport SpecialType, NONE, CLL, CLR, VS, GBJ, GBW
import numpy as np
cimport numpy as np

import sound_law.data.alphabet as alphabet

PyNoStress = <int>NOSTRESS
PyStressed = <int>STRESSED
PyUnstressed = <int>UNSTRESSED

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

    # @property
    # def dist(self) -> float:
    #     return self.ptr.dist

    # def is_leaf(self) -> bool:
    #     return self.ptr.is_leaf()

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

    # @property
    # def action_count(self):
    #     return np.asarray(self.ptr.action_count, dtype='long')

    # @property
    # def max_index(self):
    #     return self.ptr.max_index

    # @property
    # def max_action_id(self):
    #     return self.ptr.max_action_id

    # @property
    # def total_value(self):
    #     return np.asarray(self.ptr.total_value, dtype='float32')

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

    def __cinit__(self, abc_t null_id, abc_t emp_id, abc_t sot_id, abc_t eot_id, abc_t any_id, abc_t any_s_id, abc_t any_uns_id):
        self.c_obj = ActionSpaceOpt()
        self.c_obj.null_id = null_id
        self.c_obj.emp_id = emp_id
        self.c_obj.sot_id = sot_id
        self.c_obj.eot_id = eot_id
        self.c_obj.any_id = any_id
        self.c_obj.any_s_id = any_s_id
        self.c_obj.any_uns_id = any_uns_id

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

    def __cinit__(self, PyEnvOpt py_env_opt, PyActionSpaceOpt py_as_opt, PyWordSpaceOpt py_ws_opt):
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

    # def step(self, PyTreeNode node, int best_i, uai_t action_id):
    #     new_node = self.ptr.apply_action(node.ptr, best_i, action_id)
    #     tnode_cls = type(node)
    #     return wrap_node(tnode_cls, new_node)

cdef class PyMcts:
    cdef Mcts *ptr

    def __cinit__(self, PyEnv py_env, PyMctsOpt py_mcts_opt):
        self.ptr = new Mcts(py_env.ptr, py_mcts_opt.c_obj)

    def __dealloc__(self):
        del self.ptr

    # def select(self, PyTreeNode py_tnode, int num_sims, int depth_limit):
    #     cdef vector[TNptr] selected = self.ptr.select(py_tnode.ptr, num_sims, depth_limit)
    #     cdef vector[int] steps_left_vec = vector[int](selected.size())
    #     cdef size_t i
    #     for i in range(selected.size()):
    #         steps_left_vec[i] = selected[i].depth
    #     steps_left = np.asarray(steps_left_vec, dtype='long')
    #     tnode_cls = type(py_tnode)
    #     return [wrap_node(tnode_cls, node) for node in selected], steps_left

    # def backup(self, states, values):
    #     cdef size_t n = len(states)
    #     cdef vector[TNptr] states_vec = vector[TNptr](n)
    #     for i in range(n):
    #         states_vec[i] = get_ptr(states[i])
    #     cdef float[::1] np_values = np.asarray(values, dtype='float32')
    #     cdef vector[float] values_vec = np2vector(np_values)
    #     self.ptr.backup(states_vec, values_vec)

    # def play(self, PyTreeNode py_tnode):
    #     cdef uai_t action_id = self.ptr.play(py_tnode.ptr)
    #     cdef float reward = py_tnode.ptr.rewards.at(action_id)
    #     cdef TreeNode *new_node = py_tnode.ptr.neighbors.at(action_id)
    #     tnode_cls = type(py_tnode)
    #     return action_id, wrap_node(tnode_cls, new_node), reward

    # def enable_timer(self):
    #     stats.enable_timer()

    # def disable_timer(self):
    #     stats.disable_timer()

    # def show_stats(self):
    #     stats.show_stats()

    # def set_logging_options(self, int verbose_level, bool log_to_file):
    #     self.ptr.set_logging_options(verbose_level, log_to_file)

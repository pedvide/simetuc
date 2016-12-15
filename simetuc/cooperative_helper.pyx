# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:41:29 2016

@author: Pedro
"""
import itertools
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython

proc_dtype = [('i', np.uint32),
              ('k', np.uint32),
              ('l', np.uint32),
              ('d_li', np.float64),
              ('d_lk', np.float64),
              ('d_ik', np.float64)]


DTYPE_double = np.float64
ctypedef np.float64_t DTYPE_double_t
DTYPE_uint = np.uint32
ctypedef np.uint32_t DTYPE_uint_t

cdef packed struct proc_dtype_t:
    DTYPE_uint_t i, k, l
    DTYPE_double_t d_li, d_lk, d_ik

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_all_processes(np.ndarray[DTYPE_uint_t, ndim=1] indices_this,
                        np.ndarray[DTYPE_uint_t, ndim=2] indices_others_in,
                        np.ndarray[DTYPE_double_t, ndim=2] dists_others,
                        double d_max_coop, unsigned int interaction_estimate):
        '''Calculate all cooperative processes from ions indices_this to all indices_others.'''
        cdef np.ndarray[proc_dtype_t, ndim=1] processes_arr, new_rows
        cdef np.ndarray[DTYPE_uint_t, ndim=2] indices_others, pairs
        cdef np.ndarray[DTYPE_double_t, ndim=2] pairs_dist
        cdef unsigned int num, num_indices_others

        processes_arr = np.empty((interaction_estimate, ), dtype=proc_dtype)
        indices_others = np.array([row.reshape(len(row),) for row in indices_others_in], dtype=DTYPE_uint)

        # for each ion, and the other ions it interacts with
        num = 0
        # XXX: parallelize?
        num_indices_others = indices_others_in.shape[0]
        for i in range(num_indices_others):
            index_this = indices_this[i]
            indices_k = indices_others[i]
            dists_k = dists_others[i]

            # pairs of other ions
            pairs =  np.array(list(itertools.combinations(indices_k, 2)), dtype=DTYPE_uint)
            # distances from this to the pairs of others
            pairs_dist =  np.array(list(itertools.combinations(dists_k, 2)), dtype=DTYPE_double)
            for j in range(len(pairs)):
                if pairs_dist[j, 0] < d_max_coop and pairs_dist[j, 1] < d_max_coop:
                    processes_arr[i:i+1] = (pairs[j, 0], pairs[j, 1], index_this,
                                            pairs_dist[j, 0], pairs_dist[j, 1], 0.0)
        return processes_arr[0:i]



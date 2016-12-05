# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:13:09 2016

@author: Villanueva
"""

import numpy as np
cimport numpy as np
from scipy.sparse import csc_matrix

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _rate_eq_pulse_assimulo(abs_matrix, decay_matrix, UC_matrix, N_indices):
    def _rate_eq_pulse(DTYPE_t t, np.ndarray y):
        ''' Calculates the rhs of the ODE for the excitation pulse
        '''
        cdef np.ndarray N_prod_sel = y[N_indices[:, 0]]*y[N_indices[:, 1]]
        cdef np.ndarray UC_matrix_prod = UC_matrix.dot(N_prod_sel)

        return abs_matrix.dot(y) + decay_matrix.dot(y) + UC_matrix_prod

    return _rate_eq_pulse


def _jac_rate_eq_pulse_assimulo(abs_matrix, decay_matrix, UC_matrix, jac_indices):
    def _jac_rate_eq_pulse(DTYPE_t t, np.ndarray y):
        ''' Calculates the jacobian of the ODE for the excitation pulse
        '''
        cdef np.ndarray y_values = y[jac_indices[:, 2]]
        nJ_matrix = csc_matrix((y_values, (jac_indices[:, 0], jac_indices[:, 1])),
                               shape=(UC_matrix.shape[1], UC_matrix.shape[0]), dtype=np.float64)
        cdef np.ndarray UC_J_matrix = (UC_matrix.dot(nJ_matrix)).toarray()

        return abs_matrix.toarray() + decay_matrix.toarray() + UC_J_matrix

    return _jac_rate_eq_pulse


def _rate_eq_assimulo(decay_matrix, UC_matrix, N_indices):
    '''Closure with the external arguments to rate_eq'''
    def _rate_eq(DTYPE_t t, np.ndarray y):
        '''Calculates the rhs of the ODE for the relaxation'''
        cdef np.ndarray N_prod_sel = y[N_indices[:, 0]]*y[N_indices[:, 1]]
        cdef np.ndarray UC_matrix_prod = UC_matrix.dot(N_prod_sel)
        return decay_matrix.dot(y) + UC_matrix_prod

    return _rate_eq


def _jac_rate_eq_assimulo(decay_matrix, UC_matrix, jac_indices):
    '''Closure with the external arguments to rate_eq'''
    def _jac_rate_eq(DTYPE_t t, np.ndarray y):
        ''' Calculates the jacobian of the ODE for the relaxation
        '''
        cdef np.ndarray y_values = y[jac_indices[:, 2]]
        nJ_matrix = csc_matrix((y_values, (jac_indices[:, 0], jac_indices[:, 1])),
                               shape=(UC_matrix.shape[1], UC_matrix.shape[0]), dtype=np.float64)
        cdef np.ndarray UC_J_matrix = (UC_matrix.dot(nJ_matrix)).toarray()
        return UC_J_matrix + decay_matrix.toarray()

    return _jac_rate_eq

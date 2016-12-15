# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:44:51 2016

@author: Pedro
"""
# unused arguments, needed for ODE solver
# pylint: disable=W0613

import logging
import warnings
import ctypes

import numpy as np
from numpy.ctypeslib import ndpointer

from scipy.sparse import csr_matrix, csc_matrix
from scipy.integrate import ode

# nice progress bar
from tqdm import tqdm

def _rate_eq_pulse(t, y, abs_matrix, decay_matrix,
                   UC_matrix, N_indices,
                   coop_ET_matrix, coop_N_indices):
    ''' Calculates the rhs of the ODE for the excitation pulse
    '''
    N_prod_sel = y[N_indices[:, 0]]*y[N_indices[:, 1]]
    UC_matrix = UC_matrix.dot(N_prod_sel)

    N_coop_prod_sel = y[coop_N_indices[:, 0]]*y[coop_N_indices[:, 1]]*y[coop_N_indices[:, 2]]
    coop_ET_matrix = coop_ET_matrix.dot(N_coop_prod_sel)

    return abs_matrix.dot(y) + decay_matrix.dot(y) + UC_matrix + coop_ET_matrix


def _jac_rate_eq_pulse(t, y, abs_matrix, decay_matrix,
                       UC_matrix, jac_indices,
                       coop_ET_matrix, coop_jac_indices):
    ''' Calculates the jacobian of the ODE for the excitation pulse
    '''
    y_values = y[jac_indices[:, 2]]
    nJ_matrix = csr_matrix((y_values, (jac_indices[:, 0], jac_indices[:, 1])),
                           shape=(UC_matrix.shape[1], UC_matrix.shape[0]), dtype=np.float64)
    UC_J_matrix = UC_matrix.dot(nJ_matrix).toarray()

    y_coop_values = y[coop_jac_indices[:, 2]]*y[coop_jac_indices[:, 3]]
    nJ_coop_matrix = csr_matrix((y_coop_values, (coop_jac_indices[:, 0], coop_jac_indices[:, 1])),
                           shape=(coop_ET_matrix.shape[1], coop_ET_matrix.shape[0]),
                           dtype=np.float64)
    UC_J_coop_matrix = coop_ET_matrix.dot(nJ_coop_matrix).toarray()

    return abs_matrix.toarray() + decay_matrix.toarray() + UC_J_matrix + UC_J_coop_matrix


def _rate_eq(t, y, decay_matrix, UC_matrix, N_indices, coop_ET_matrix, coop_N_indices):
    '''Calculates the rhs of the ODE for the relaxation'''
    N_prod_sel = y[N_indices[:, 0]]*y[N_indices[:, 1]]
    UC_matrix = UC_matrix.dot(N_prod_sel)

    N_coop_prod_sel = y[coop_N_indices[:, 0]]*y[coop_N_indices[:, 1]]*y[coop_N_indices[:, 2]]
    coop_ET_matrix = coop_ET_matrix.dot(N_coop_prod_sel)

    return decay_matrix.dot(y) + UC_matrix + coop_ET_matrix


def _jac_rate_eq(t, y, decay_matrix, UC_matrix, jac_indices, coop_ET_matrix, coop_jac_indices):
    ''' Calculates the jacobian of the ODE for the relaxation
    '''
    y_values = y[jac_indices[:, 2]]
    nJ_matrix = csr_matrix((y_values, (jac_indices[:, 0], jac_indices[:, 1])),
                           shape=(UC_matrix.shape[1], UC_matrix.shape[0]), dtype=np.float64)
    UC_J_matrix = UC_matrix.dot(nJ_matrix).toarray()

    y_coop_values = y[coop_jac_indices[:, 2]]*y[coop_jac_indices[:, 3]]
    nJ_coop_matrix = csr_matrix((y_coop_values, (coop_jac_indices[:, 0], coop_jac_indices[:, 1])),
                           shape=(coop_ET_matrix.shape[1], coop_ET_matrix.shape[0]),
                           dtype=np.float64)
    UC_J_coop_matrix = coop_ET_matrix.dot(nJ_coop_matrix).toarray()

    return decay_matrix.toarray() + UC_J_matrix + UC_J_coop_matrix


def _rate_eq_dll(decay_matrix, UC_matrix, N_indices):  # pragma: no cover
    ''' Calculates the rhs of the ODE for the relaxation using odesolver.dll'''
    ext_odesolver = ctypes.windll.odesolver

    matrix_ctype = ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS')
    # uint64 not supported by c++?, use int32
    vector_int_ctype = ndpointer(dtype=np.int32, ndim=1, flags='F_CONTIGUOUS')
    vector_ctype = ndpointer(dtype=np.float64, ndim=1, flags='F_CONTIGUOUS')

    ext_odesolver.rateEq.argtypes = [ctypes.c_double, vector_ctype,
                                     matrix_ctype,
                                     vector_ctype, vector_int_ctype, vector_int_ctype,
                                     vector_int_ctype, vector_int_ctype,
                                     ctypes.c_uint, ctypes.c_uint, vector_ctype]
    ext_odesolver.rateEq.restype = ctypes.c_int

    n_states = decay_matrix.shape[0]
    n_inter = UC_matrix.shape[1]

#        initial_population = np.asfortranarray(initial_population, dtype=np.float64)

    # eigen uses Fortran ordering
#    abs_matrix = np.asfortranarray(abs_matrix.toarray(), dtype=np.float64)
    decay_matrix = np.asfortranarray(decay_matrix.toarray(), dtype=np.float64)

    # precalculate gives csr matrix, which isn't fortran style
    UC_matrix = csc_matrix(UC_matrix)  # precalculate gives csr matrix, which isn't fortran style
    UC_matrix_data = np.asfortranarray(UC_matrix.data, dtype=np.float64)
    UC_matrix_indices = np.asfortranarray(UC_matrix.indices, dtype=np.uint32)
    UC_matrix_indptr = np.asfortranarray(UC_matrix.indptr, dtype=np.uint32)

    N_indices_i = np.asfortranarray(N_indices[:, 0], dtype=np.uint32)
    N_indices_j = np.asfortranarray(N_indices[:, 1], dtype=np.uint32)

    out_vector = np.asfortranarray(np.zeros((n_states,)), dtype=np.float64)

    def rate_eq_fast(t, y):
        '''Calculates the rhs of the ODE for the relaxation'''
        ext_odesolver.rateEq(y,
                             decay_matrix,
                             UC_matrix_data, UC_matrix_indices, UC_matrix_indptr,
                             N_indices_i, N_indices_j,
                             n_states, n_inter, out_vector)
        return out_vector

    return rate_eq_fast


def _solve_ode(t_arr, fun, fargs, jfun, jargs, initial_population,
               rtol=1e-3, atol=1e-15, nsteps=500, method='bdf', quiet=True):
    ''' Solve the ode for the times t_arr using rhs fun and jac jfun
        with their arguments as tuples.
    '''
    logger = logging.getLogger(__name__)

    N_steps = len(t_arr)
    y_arr = np.zeros((N_steps, len(initial_population)), dtype=np.float64)

    # setup the ode solver with the method
    ode_obj = ode(fun, jfun)
    ode_obj.set_integrator('vode', rtol=rtol, atol=atol, method=method, nsteps=nsteps)
    ode_obj.set_initial_value(initial_population, t_arr[0])
    ode_obj.set_f_params(*fargs)
    ode_obj.set_jac_params(*jargs)

    # initial conditions
    y_arr[0, :] = initial_population
    step = 1

    # console bar enabled for INFO
    # this doesn't work, as there are two handlers with different levels
    cmd_bar_disable = quiet
    pbar_cmd = tqdm(total=N_steps, unit='step', smoothing=0.1,
                    disable=cmd_bar_disable, desc='ODE progress')

    # catch numpy warnings and log them
    # DVODE (the internal routine used by the integrator 'vode') will throw a warning
    # if it needs too many steps to solve the ode.
    np.seterr(all='raise')
    with warnings.catch_warnings():
        # transform warnings into exceptions that we can catch
        warnings.filterwarnings('error')
        try:
            while ode_obj.successful() and step < N_steps:
                # advance ode to the next time step
                y_arr[step, :] = ode_obj.integrate(t_arr[step])
                step += 1
                pbar_cmd.update(1)
        except UserWarning as err:  # pragma: no cover
            logger.warning(str(err))
            logger.warning('Most likely the ode solver is taking too many steps.')
            logger.warning('Either change your settings or increase "nsteps".')
            logger.warning('The program will continue, but the accuracy of the ' +
                           'results cannot be guaranteed.')
    np.seterr(all='ignore')  # restore settings

    pbar_cmd.update(1)
    pbar_cmd.close()

    return y_arr


def solve_pulse(t_pulse, initial_pop, total_abs_matrix, decay_matrix,
                UC_matrix, N_indices, jac_indices,
                coop_ET_matrix, coop_N_indices, coop_jac_indices,
                nsteps=100, rtol=1e-3, atol=1e-15, quiet=False):
    '''Solve the response to an excitation pulse.'''
    return _solve_ode(t_pulse, _rate_eq_pulse,
                      (total_abs_matrix, decay_matrix, UC_matrix, N_indices,
                       coop_ET_matrix, coop_N_indices),
                      _jac_rate_eq_pulse,
                      (total_abs_matrix, decay_matrix, UC_matrix, jac_indices,
                       coop_ET_matrix, coop_jac_indices),
                      initial_pop, method='adams',
                      rtol=rtol, atol=atol, nsteps=nsteps, quiet=quiet)


def solve_relax(t_sol, initial_pop, decay_matrix, UC_matrix, N_indices, jac_indices,
                coop_ET_matrix, coop_N_indices, coop_jac_indices,
                nsteps=1000, rtol=1e-3, atol=1e-15, quiet=False):
    '''Solve the relaxation after a pulse.'''
    return _solve_ode(t_sol, _rate_eq,
                      (decay_matrix, UC_matrix, N_indices, coop_ET_matrix, coop_N_indices),
                      _jac_rate_eq,
                      (decay_matrix, UC_matrix, jac_indices, coop_ET_matrix, coop_jac_indices),
                      initial_pop, rtol=rtol, atol=atol,
                      nsteps=nsteps, quiet=quiet)

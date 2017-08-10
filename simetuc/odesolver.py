# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:44:51 2016

@author: Pedro
"""

# unused arguments, needed for ODE solver
# pylint: disable=W0613

import logging
import warnings

from typing import Callable, Tuple

import numpy as np

from scipy.sparse import csr_matrix
from scipy.integrate import ode

# nice progress bar
from tqdm import tqdm


def _rate_eq(t: np.array, y: np.array, decay_matrix: np.array,
             UC_matrix: np.array, N_indices: np.array,
             coop_ET_matrix: np.array, coop_N_indices: np.array) -> np.array:
    '''Calculates the rhs of the ODE for the relaxation'''
    N_prod_sel = y[N_indices[:, 0]]*y[N_indices[:, 1]]
    UC_matrix = UC_matrix.dot(N_prod_sel)

    N_coop_prod_sel = y[coop_N_indices[:, 0]]*y[coop_N_indices[:, 1]]*y[coop_N_indices[:, 2]]
    coop_ET_matrix = coop_ET_matrix.dot(N_coop_prod_sel)

    return decay_matrix.dot(y) + UC_matrix + coop_ET_matrix


def _jac_rate_eq(t: np.array, y: np.array, decay_matrix: np.array,
                 UC_matrix: np.array, jac_indices: np.array,
                 coop_ET_matrix: np.array, coop_jac_indices: np.array) -> np.array:
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


def _rate_eq_pulse(t: np.array, y: np.array, abs_matrix: np.array, decay_matrix: np.array,
                   UC_matrix: np.array, N_indices: np.array,
                   coop_ET_matrix: np.array, coop_N_indices: np.array) -> np.array:
    ''' Calculates the rhs of the ODE for the excitation pulse
    '''
    return abs_matrix.dot(y) + _rate_eq(t, y, decay_matrix, UC_matrix, N_indices,
                                        coop_ET_matrix, coop_N_indices)


def _jac_rate_eq_pulse(t: np.array, y: np.array, abs_matrix: np.array, decay_matrix: np.array,
                       UC_matrix: np.array, jac_indices: np.array,
                       coop_ET_matrix: np.array, coop_jac_indices: np.array) -> np.array:
    ''' Calculates the jacobian of the ODE for the excitation pulse
    '''
    return abs_matrix.toarray() + _jac_rate_eq(t, y, decay_matrix, UC_matrix, jac_indices,
                                               coop_ET_matrix, coop_jac_indices)




def _solve_ode(t_arr: np.array,
               fun: Callable, fargs: Tuple,
               jfun: Callable, jargs: Tuple,
               initial_population: np.array,
               rtol: float = 1e-3, atol: float = 1e-15, nsteps: int = 500,
               method: str = 'bdf', quiet: bool = True) -> np.array:
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

    # catch numpy warnings and log them
    # DVODE (the internal routine used by the integrator 'vode') will throw a warning
    # if it needs too many steps to solve the ode.
    with warnings.catch_warnings(), np.errstate(all='raise'), tqdm(total=N_steps, unit='step',
                                                                   smoothing=0.1,
                                                                   disable=cmd_bar_disable,
                                                                   desc='ODE progress') as pbar_cmd:
        # transform warnings into exceptions that we can catch
        warnings.filterwarnings('error')
        while ode_obj.successful() and step < N_steps:
            try:
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
        pbar_cmd.update(1)

    return y_arr


def solve_pulse(t_pulse: np.array, initial_pop: np.array,
                total_abs_matrix: csr_matrix, decay_matrix: csr_matrix,
                UC_matrix: csr_matrix, N_indices: np.array, jac_indices: np.array,
                coop_ET_matrix: csr_matrix,
                coop_N_indices: np.array, coop_jac_indices: np.array,
                nsteps: int = 100, rtol: float = 1e-3, atol: float = 1e-15,
                quiet: bool = False, method: str = 'adams') -> np.array:
    '''Solve the response to an excitation pulse.'''
    return _solve_ode(t_pulse, _rate_eq_pulse,
                      (total_abs_matrix, decay_matrix, UC_matrix, N_indices,
                       coop_ET_matrix, coop_N_indices),
                      _jac_rate_eq_pulse,
                      (total_abs_matrix, decay_matrix, UC_matrix, jac_indices,
                       coop_ET_matrix, coop_jac_indices),
                      initial_pop, method=method,
                      rtol=rtol, atol=atol, nsteps=nsteps, quiet=quiet)


def solve_relax(t_sol: np.array, initial_pop: np.array,
                decay_matrix: csr_matrix,
                UC_matrix: csr_matrix, N_indices: np.array, jac_indices: np.array,
                coop_ET_matrix: csr_matrix,
                coop_N_indices: np.array, coop_jac_indices: np.array,
                nsteps: int = 1000, rtol: float = 1e-3, atol: float = 1e-15,
                quiet: bool = False) -> np.array:
    '''Solve the relaxation after a pulse.'''
    return _solve_ode(t_sol, _rate_eq,
                      (decay_matrix, UC_matrix, N_indices, coop_ET_matrix, coop_N_indices),
                      _jac_rate_eq,
                      (decay_matrix, UC_matrix, jac_indices, coop_ET_matrix, coop_jac_indices),
                      initial_pop, rtol=rtol, atol=atol,
                      nsteps=nsteps, quiet=quiet)

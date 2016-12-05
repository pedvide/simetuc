# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:44:51 2016

@author: Pedro
"""
# XXX: This is actually much slower than the odesolve using scipy....
# I can't compile it in windows, so I've used conda install assimulo,
# maybe if it was compiled it'd be faster?

# unused arguments, needed for ODE solver
# pylint: disable=W0613
#import numpy as np

#from scipy.sparse import csc_matrix

from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

import numpy
import pyximport
pyximport.install(setup_args={"script_args":["--compiler=msvc"],
                              "include_dirs": numpy.get_include()},
                  reload_support=True)
from odesolver_assimulo_funs import _rate_eq_pulse_assimulo, _jac_rate_eq_pulse_assimulo
from odesolver_assimulo_funs import _rate_eq_assimulo, _jac_rate_eq_assimulo


def _solve_ode_assimulo(t_arr, fun, fargs, jfun, jargs, initial_population,
                        rtol=1e-3, atol=1e-15, nsteps=500, method='bdf', quiet=True):

    f = fun(*fargs)
    jac = jfun(*jargs)

    exp_mod = Explicit_Problem(f, initial_population, t_arr[0])
    exp_mod.jac = jac #Sets the Jacobian

    exp_sim = CVode(exp_mod) #Create a CVode solver
    #Set the parameters
    exp_sim.iter = 'Newton' #Default 'FixedPoint'
    exp_sim.discr = method
    exp_sim.atol = atol
    exp_sim.rtol = rtol

    #Simulate
    t, y_arr = exp_sim.simulate(t_arr[-1], 0, t_arr)

    return y_arr


def solve_relax(t_sol, initial_pop, decay_matrix, UC_matrix, N_indices, jac_indices,
                nsteps=1000, rtol=1e-3, atol=1e-15, quiet=False):
    '''Solve the relaxation after a pulse.'''
    return _solve_ode_assimulo(t_sol, _rate_eq_assimulo,
                               (decay_matrix, UC_matrix, N_indices),
                               _jac_rate_eq_assimulo,
                               (decay_matrix, UC_matrix, jac_indices),
                               initial_pop, rtol=rtol, atol=atol,
                               nsteps=nsteps, quiet=quiet)


def solve_pulse(t_pulse, initial_pop, total_abs_matrix, decay_matrix,
                UC_matrix, N_indices, jac_indices,
                nsteps=100, rtol=1e-3, atol=1e-15, quiet=False):
    '''Solve the response to an excitation pulse.'''
    return _solve_ode_assimulo(t_pulse, _rate_eq_pulse_assimulo,
                               (total_abs_matrix, decay_matrix, UC_matrix, N_indices),
                               _jac_rate_eq_pulse_assimulo,
                               (total_abs_matrix, decay_matrix, UC_matrix, jac_indices),
                               initial_pop, method='adams',
                               rtol=rtol, atol=atol, nsteps=nsteps, quiet=quiet)

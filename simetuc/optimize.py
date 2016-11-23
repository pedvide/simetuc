# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:58:58 2016

@author: Pedro
"""

import time
import logging
import functools
from typing import Dict, Tuple

import numpy as np
# pylint: disable=E1101
# pylint: disable=W0613
import scipy.optimize as optimize

import tqdm

import simetuc.simulations as simulations


# I can't seem to find the right mypy syntax for this decorator.
def cache(function):
    '''Decorator to store a list of parameters and function values'''
    cache.params_lst = []
    cache.f_val_lst = []

    @functools.wraps(function)
    def wrapper(*args):
        '''Wraps a function to add a cache'''
        f_val = function(*args)
        cache.params_lst.append(tuple(*args))
        cache.f_val_lst.append(f_val)
        return f_val
    return wrapper


def optimize_dynamics(cte: Dict) -> Tuple[float, float, float]:
    ''' Minimize the error between experimental data and simulation for the settings in cte
    '''
    logger = logging.getLogger(__name__)

    def callback_fun(Xi):
        ''' This function is called after every minimization step
            It prints the current parameters and error from the cache
        '''
        pbar.update(1)
        if not cte['no_console']:
            format_params = ', '.join('{:.3e}'.format(val) for val in cache.params_lst[-1]*x0)
            msg = '({}): {:.3e}'.format(format_params, cache.f_val_lst[-1])
            tqdm.tqdm.write(msg)

    def _update_ET_and_simulate(x):
        # update ET values if explicitly given
        for num, process in enumerate(process_list):
            sim.modify_ET_param_value(process, x[num]*x0[num])  # precondition

        dynamics_sol = sim.simulate_dynamics()
        total_error = dynamics_sol.total_error

        return total_error

    @cache
    def fun_optim(x: np.array) -> float:
        ''' Function to optimize.
            The parameters and results are stored in the cache decorator
        '''
        total_error = _update_ET_and_simulate(x)
        return total_error

    def fun_optim_brute(x: np.array) -> float:
        ''' Function to optimize by brute force.
        '''
        pbar.update(1)
        total_error = _update_ET_and_simulate(x)
        return total_error

    cte['no_plot'] = True

    sim = simulations.Simulations(cte)

    start_time = time.time()

    # read the starting values from the settings
    if 'optimization_processes' in cte and cte['optimization_processes'] is not None:
        # optimize only those ET parameters that the user has selected
        process_list = np.array([process for process in cte['ET']
                                 if process in cte['optimization_processes']])
    else:
        process_list = np.array([process for process in cte['ET']])

    # starting point
    x0 = np.array([cte['ET'][process]['value'] for process in process_list])

    tol = 1e-4
    bounds = ((0, 1e10),)*len(x0)

    method = 'SLSQP'

    if method == 'COBYLA':
        # minimize error. The starting point is preconditioned to be 1
        res = optimize.minimize(fun_optim, np.ones_like(x0), method=method, tol=tol)

    if method == 'L-BFGS-B':  # fails?
        pbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
        logger.info('ET parameters. RMSD.')
        res = optimize.minimize(fun_optim, np.ones_like(x0), bounds=bounds,
                                method=method, tol=tol, callback=callback_fun)
        pbar.close()

    if method == 'TNC':  # fails?
        pbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
        logger.info('ET parameters. RMSD.')
        res = optimize.minimize(fun_optim, np.ones_like(x0), bounds=bounds,
                                method=method, tol=tol, callback=callback_fun)
        pbar.close()

    if method == 'SLSQP':
        pbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
        logger.info('ET parameters. RMSD.')
        res = optimize.minimize(fun_optim, np.ones_like(x0), bounds=bounds,
                                method=method, tol=tol, callback=callback_fun)
        pbar.close()

    if method == 'basin_hopping':
        minimizer_kwargs = {"method": "SLSQP"}

#        def accept_test(f_new, x_new, f_old, x_old) :
#            return np.alltrue(x_new > 0)

        pbar = tqdm.tqdm(desc='Optimizing', unit=' points', disable=cte['no_console'])
        logger.info('ET parameters. RMSD.')
        res = optimize.basinhopping(fun_optim, np.ones_like(x0), minimizer_kwargs=minimizer_kwargs,
                                    niter=10, stepsize=5, T=1e-2, callback=callback_fun)
        pbar.close()

    if method == 'brute':
        rranges = ((1e-2, 6),)*len(x0)
        N_points = 50
        pbar = tqdm.tqdm(desc='Optimizing', total=1+N_points**2,
                         unit='points', disable=cte['no_console'])
        res = optimize.brute(fun_optim_brute, ranges=rranges, Ns=N_points, full_output=True,
                             finish=None, disp=True)
        pbar.close()
        best_x = res[0]*x0
        min_f = res[1]

    if method != 'brute':
        best_x = res.x*x0
        min_f = res.fun

    print('\n\n')
    logger.info(res)

#    lattice_name = cte['lattice']['name']
#    path = os.path.join('results', lattice_name, 'minimize_results')
#    np.savez(path, res=res, cte=cte, best_x=best_x)  # save results

    total_time = time.time()-start_time
    formatted_time = time.strftime("%Hh %Mm %Ss", time.localtime(total_time))
    logger.info('Minimum reached! Total time: %s.', formatted_time)
    logger.info('Minimum value: %s', np.array_str(np.array(min_f), precision=5))

    return (best_x, min_f, total_time)


#if __name__ == "__main__":
#    logger = logging.getLogger()
#    logging.basicConfig(level=logging.WARNING,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#
#    logger.debug('Called from cmd.')
#
#    import simetuc.settings as settings
#    cte = settings.load('config_file.cfg')
#
#    cte['no_console'] = False
#    cte['no_plot'] = True
#
#    optimize_dynamics(cte)

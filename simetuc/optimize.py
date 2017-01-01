# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:58:58 2016

@author: Pedro
"""

import datetime
import logging
import functools
from typing import Dict, Tuple, List, Callable

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

def optim_fun_factory(sim: simulations.Simulations,
                      process_list: List[str], x0: np.array,
                      average: bool = False) -> Callable:
    '''Generate the function to be optimize.
        This function modifies the ET params and return the error
    '''
    def optim_fun(x):
        # update ET values if explicitly given
        for num, process in enumerate(process_list):
            sim.modify_ET_param_value(process, x[num]*x0[num])  # precondition

        dynamics_sol = sim.simulate_dynamics(average=average)
        total_error = dynamics_sol.total_error

        return total_error

    return optim_fun


def optimize_dynamics(cte: Dict, method: str = None,
                      average: bool = False) -> Tuple[np.array, float]:
    ''' Minimize the error between experimental data and simulation for the settings in cte
        average = True -> optimize average rate equations instead of microscopic ones
    '''
    logger = logging.getLogger(__name__)

    def callback_fun(Xi, *args):
        ''' This function is called after every minimization step
            It prints the current parameters and error from the cache
        '''
        pbar.update(1)
        if not cte['no_console']:
            format_params = ', '.join('{:.3e}'.format(val)
                                      for val in cache.params_lst[-1]*x0)
            msg = '({}): {:.3e}'.format(format_params, cache.f_val_lst[-1])
            tqdm.tqdm.write(msg)

    cte['no_plot'] = True

    sim = simulations.Simulations(cte)

    start_time = datetime.datetime.now()

    # read the starting values from the settings
    if 'optimization_processes' in cte and cte['optimization_processes'] is not None:
        # optimize only those ET parameters that the user has selected
        process_list = [process for process in cte['ET']
                        if process in cte['optimization_processes']]
    else:
        process_list = [process for process in cte['ET']]

    # starting point
    # use the avg value if present
    ET_dict = cte['ET'].copy()
    if average:
        for dict_process in ET_dict.values():
            if 'value_avg' in dict_process:
                dict_process['value'] = dict_process['value_avg']
    x0 = np.array([ET_dict[process]['value'] for process in process_list])

    tol = 1e-4
    bounds = ((0, 1e10),)*len(x0)

    if method is None:
        method = 'SLSQP'
    logger.info('Optimization method: %s.', method)

    _optim_fun = optim_fun_factory(sim, process_list, x0, average=average)
    if method != 'brute_force':
        _optim_fun = cache(_optim_fun)

    if method == 'COBYLA':
        # minimize error. The starting point is preconditioned to be 1
        res = optimize.minimize(_optim_fun, np.ones_like(x0),
                                method=method, tol=tol)

    elif method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP':
        pbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
        logger.info('ET parameters. RMSD.')
        res = optimize.minimize(_optim_fun, np.ones_like(x0), bounds=bounds,
                                method=method, tol=tol, callback=callback_fun)
        pbar.close()

    elif method == 'basin_hopping':
        minimizer_kwargs = {"method": "SLSQP"}

#        def accept_test(f_new, x_new, f_old, x_old) :
#            return np.alltrue(x_new > 0)

        pbar = tqdm.tqdm(desc='Optimizing', unit=' points', disable=cte['no_console'])
        logger.info('ET parameters. RMSD.')
        res = optimize.basinhopping(_optim_fun, np.ones_like(x0),
                                    minimizer_kwargs=minimizer_kwargs,
                                    niter=10, stepsize=5, T=1e-2, callback=callback_fun)
        pbar.close()

    elif method == 'brute_force':
        rranges = ((1e-2, 6),)*len(x0)
        N_points = 50
        pbar = tqdm.tqdm(desc='Optimizing', total=1+N_points**2,
                         unit='points', disable=cte['no_console'])
        res = optimize.brute(_optim_fun, ranges=rranges, Ns=N_points, full_output=True,
                             finish=None, disp=True)
        pbar.close()
        best_x = res[0]*x0
        min_f = res[1]

    if method != 'brute_force':
        best_x = res.x*x0
        min_f = res.fun

    logger.debug(res)

    total_time = datetime.datetime.now() - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = '{:.0f}h {:02.0f}m {:02.0f}s'.format(hours, minutes, seconds)
    logger.info('Minimum reached! Total time: %s.', formatted_time)
    logger.info('Minimum value: %s', np.array_str(np.array(min_f), precision=5))

    return best_x, min_f


if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    logger.debug('Called from cmd.')

    import simetuc.settings as settings
    cte = settings.load('config_file.cfg')

    cte['no_console'] = False
    cte['no_plot'] = True

    optimize_dynamics(cte, average=False)

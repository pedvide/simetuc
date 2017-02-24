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
    # if @cache mypy syntax is fixed, add annotations here
    def wrapper(*args):
        '''Wraps a function to add a cache'''
        f_val = function(*args)
        cache.params_lst.append(tuple(*args))
        cache.f_val_lst.append(f_val)
        return f_val
    return wrapper


def optim_fun_factory(sim: simulations.Simulations,
                      process_list: List[str], x0: np.array,
                      average: bool = False, pbar: tqdm.tqdm = None) -> Callable:
    '''Generate the function to be optimized.
        This function modifies the ET params and return the error
    '''
    def optim_fun(x: np.array) -> float:
        '''Update ET strengths, simulate dynamics and return total error'''
        # update ET values if explicitly given
        for num, process in enumerate(process_list):
            sim.modify_param_value(process, x[num]*x0[num])  # precondition

        dynamics_sol = sim.simulate_dynamics(average=average)
        total_error = dynamics_sol.total_error
        # if a progress bar is given, advance it
        if pbar is not None:
            pbar.update(1)

        return total_error

    return optim_fun


def optimize_dynamics(cte: Dict, method: str = None,
                      average: bool = False) -> Tuple[np.array, float]:
    ''' Minimize the error between experimental data and simulation for the settings in cte
        average = True -> optimize average rate equations instead of microscopic ones
    '''
    logger = logging.getLogger(__name__)

    from simetuc.util import change_console_logger_level as change_console_logger_level
    from simetuc.util import get_console_logger_level as get_console_logger_level
    old_level = get_console_logger_level()
    # disable logging from other modules
#    logging.getLogger('simetuc.simulations').setLevel(logging.WARNING)
#    logging.getLogger('simetuc.precalculate').setLevel(logging.WARNING)
#    logging.getLogger('simetuc.lattice').setLevel(logging.WARNING)

    # if @cache mypy syntax is fixed, add annotations here
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

    # Processes to optimize. If not given, all ET parameters will be optimized
    process_list = cte.get('optimization_processes', cte['ET'])

    # use the avg value if present
    ET_dict = cte['ET'].copy()
    if average:
        for dict_process in ET_dict.values():
            if 'value_avg' in dict_process:
                dict_process['value'] = dict_process['value_avg']

    # starting point
    x0 = np.array([ET_dict[process]['value'] if isinstance(process, str) else sim.get_branching_ratio_value(process)
                   for process in process_list])

    tol = 1e-4
    # the bounds depend on the type of process
    # the bound are then normalized to x0
    max_values = np.array([1e20 if isinstance(process, str) else 1 for process in process_list])
    bounds = [(0, max_val) for max_val in max_values]
    print(max_values)
    print(bounds)

    # select optimization method
    if method is None:
        method = 'SLSQP'
    logger.info('Optimization method: %s.', method)

    # use the cache and print it if the method isn't brute force
    if method != 'brute_force':
        _optim_fun = optim_fun_factory(sim, process_list, x0, average=average)
        _optim_fun = cache(_optim_fun)
        msg = '(' + ', '.join('{}'.format(proc) for proc in process_list)
        tqdm.tqdm.write(msg + '): RMS Error')
        change_console_logger_level(logging.WARNING)

        if method == 'COBYLA':
            # minimize error. The starting point is preconditioned to be 1
            res = optimize.minimize(_optim_fun, np.ones_like(x0),
                                    method=method, tol=tol)

        elif method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP':
            pbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
            res = optimize.minimize(_optim_fun, np.ones_like(x0), bounds=bounds,
                                    method=method, tol=tol, callback=callback_fun)
            pbar.update(1)
            pbar.close()

        elif method == 'basin_hopping':
            minimizer_kwargs = {"method": "SLSQP"}

    #        def accept_test(f_new, x_new, f_old, x_old) :
    #            return np.alltrue(x_new > 0)

            pbar = tqdm.tqdm(desc='Optimizing', unit=' points', disable=cte['no_console'])
            res = optimize.basinhopping(_optim_fun, np.ones_like(x0),
                                        minimizer_kwargs=minimizer_kwargs,
                                        niter=10, stepsize=5, T=1e-2, callback=callback_fun)
            pbar.update(1)
            pbar.close()

        best_x = res.x*x0
        min_f = res.fun

    elif method == 'brute_force':
        # range and number of points. Total number is 1+N_points**2
        rranges = ((1e-1, 10),)*len(x0)
        N_points = 10
        change_console_logger_level(logging.WARNING)
        pbar = tqdm.tqdm(desc='Optimizing', total=1+N_points**2,
                         unit='points', disable=cte['no_console'])

        _optim_fun = optim_fun_factory(sim, process_list, x0, average=average, pbar=pbar)
        res = optimize.brute(_optim_fun, ranges=rranges, Ns=N_points, full_output=True,
                             finish=None, disp=True)
        pbar.update(1)
        pbar.close()
        best_x = res[0]*x0
        min_f = res[1]

    else:
        msg = 'Wrong optimization method!'
        logger.error(msg)
        raise ValueError(msg)

    change_console_logger_level(old_level)

    logger.debug(res)

    total_time = datetime.datetime.now() - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print('')
    formatted_time = '{:.0f}h {:02.0f}m {:02.0f}s'.format(hours, minutes, seconds)
    logger.info('Minimum reached! Total time: %s.', formatted_time)
    logger.info('Optimized RMS error: %.3e.', min_f)
    logger.info('Parameters name and value:')
    for proc, best_val in zip(process_list, best_x.T):
        logger.info('%s: %.3e.', proc, best_val)

    return best_x, min_f


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
#    optimize_dynamics(cte, average=False)

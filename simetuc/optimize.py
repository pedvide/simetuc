# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:58:58 2016

@author: Pedro
"""

import datetime
import logging
import functools
from typing import Tuple, List, Callable, Union

import numpy as np
# pylint: disable=E1101
# pylint: disable=W0613
import scipy.optimize as optimize

import tqdm
import h5py
import ruamel_yaml as yaml

import simetuc.simulations as simulations
import simetuc.settings as settings
import simetuc.plotter as plotter
from simetuc.util import change_console_logger_level, get_console_logger_level, Transition


# I can't seem to find the right mypy syntax for this decorator.
def cache(function):  # type: ignore
    '''Decorator to store a list of parameters and function values'''
    cache.params_lst = []
    cache.f_val_lst = []

    # if @cache mypy syntax is fixed, add annotations here
    @functools.wraps(function)
    def wrapper(*args):  # type: ignore
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
        This function modifies the ET params and returns the total error.
    '''
    if not sim.cte.optimization.get('excitations', False):
        def optim_fun(x: np.array) -> float:
            '''Update ET strengths, simulate dynamics and return total error'''
            # update ET values if explicitly given
            for process, value in zip(process_list, x*x0):
                sim.modify_param_value(process, value)  # precondition

            dynamics_sol = sim.simulate_dynamics(average=average)
            total_error = dynamics_sol.total_error
            # if a progress bar is given, advance it
            if pbar is not None:  # pragma: no cover
                pbar.update(1)
            return total_error
        return optim_fun

    else:
        # switch off all excitations
        for exc_lst in sim.cte['excitations'].values():
            for excitation in exc_lst:
                excitation.active = False

        def optim_fun_all_exc(x: np.array) -> float:
            '''Update ET strengths, simulate dynamics for all excitations and return total error.'''
            # update ET values if explicitly given
            for process, value in zip(process_list, x*x0):
                sim.modify_param_value(process, value)  # precondition

            total_error = 0.0
            # go through all required excitations, calculate errors and add all of them
            for exc_label in sim.cte.optimization['excitations']:
                # switch on one excitation, solve and switch off again
                sim.cte.excitations[exc_label][0].active = True
                dynamics_sol = sim.simulate_dynamics(average=average)
                sim.cte.excitations[exc_label][0].active = False
                total_error += dynamics_sol.total_error**2

            # if a progress bar is given, advance it
            if pbar is not None:  # pragma: no cover
                pbar.update(1)
            return np.sqrt(total_error)
        return optim_fun_all_exc


def optimize_dynamics(cte: settings.Settings, average: bool = False) -> Tuple[np.array, float]:
    ''' Minimize the error between experimental data and simulation for the settings in cte
        average = True -> optimize average rate equations instead of microscopic ones
    '''
    logger = logging.getLogger(__name__)

    # disable logging to the console.
    old_level = get_console_logger_level()

    # if @cache mypy syntax is fixed, add annotations here
    def callback_fun(Xi, *args):  # type: ignore
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
    process_list = cte.get('optimization', {}).get('processes', cte['energy_transfer'])
    logger.info('Optimization parameters: ' + ', '.join(str(proc) for proc in process_list) + '.')
#    print(process_list)

    # starting point
    x0 = np.array([sim.get_ET_param_value(process, average) if isinstance(process, str)
                      else sim.get_branching_ratio_value(process)
                   for process in process_list], dtype=np.float64)

    tol = cte.optimization.get('options', {}).get('tol', 1e-4)
    # the bounds depend on the type of process
    # the bound are then normalized to x0
    max_values = np.array([1e20 if isinstance(process, str)
                               else 1/sim.get_branching_ratio_value(process)
                           for process in process_list])
    bounds = [(0, max_val) for max_val in max_values]
#    print(bounds)

    # select optimization method
    method = cte.get('optimization', {}).get('method', 'SLSQP')
    if method is None:
        method = 'SLSQP'
    logger.info('Optimization method: %s.', method)

    # use the cache and print it if the method isn't brute force
    if method != 'brute_force':
        optim_fun = optim_fun_factory(sim, process_list, x0, average=average)
        optim_fun = cache(optim_fun)
        msg = '(' + ', '.join('{}'.format(proc) for proc in process_list)
        tqdm.tqdm.write(msg + '): RMS Error')
        change_console_logger_level(logging.WARNING)

        if method == 'COBYLA':
            # minimize error. The starting point is preconditioned to be 1
            pbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
            res = optimize.minimize(optim_fun, np.ones_like(x0),
                                    method=method, tol=tol)

        elif method in ['L-BFGS-B', 'TNC', 'SLSQP']:
            pbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
            res = optimize.minimize(optim_fun, np.ones_like(x0), bounds=bounds,
                                    method=method, tol=tol, callback=callback_fun)
            pbar.update(1)
            pbar.close()

        elif method == 'basin_hopping':
            minimizer_kwargs = {"method": "SLSQP"}

    #        def accept_test(f_new, x_new, f_old, x_old) :
    #            return np.alltrue(x_new > 0)

            pbar = tqdm.tqdm(desc='Optimizing', unit=' points', disable=cte['no_console'])
            res = optimize.basinhopping(optim_fun, np.ones_like(x0),
                                        minimizer_kwargs=minimizer_kwargs,
                                        niter=10, stepsize=5, T=1e-2, callback=callback_fun)
            pbar.update(1)
            pbar.close()
        else:
            msg = 'Wrong optimization method!'
            logger.error(msg)
            raise ValueError(msg)

        best_x = res.x*x0
        min_f = res.fun

        # save results
        _save_results(sim, process_list, [best_x], [min_f])

    else: # method == 'brute_force'
        # range and number of points. Total number is 1+N_points^(num_params)
        num_params = len(x0)
        min_range = cte.optimization.get('options', {}).get('min_bound', 1e-2)
        max_range = cte.optimization.get('options', {}).get('max_bound', 1e2)
        rranges_unbounded = ((min_range, max_range),)*num_params
        rranges = [(min_v, max_v if max_v < bound else bound)
                   for (min_v, max_v), bound in zip(rranges_unbounded, max_values)]
#        print(rranges)

        N_points = cte.optimization.get('options', {}).get('N_points', 10)
        change_console_logger_level(logging.WARNING)
        pbar = tqdm.tqdm(desc='Optimizing', total=1+N_points**num_params,
                         unit='points', disable=cte['no_console'])

        optim_fun = optim_fun_factory(sim, process_list, x0, average=average, pbar=pbar)
        res = optimize.brute(optim_fun, ranges=rranges, Ns=N_points, full_output=True,
                             finish=None, disp=True)
        pbar.update(1)
        pbar.close()
        best_x = res[0]*x0
        min_f = res[1]

        param_values = np.vstack(row.ravel() for row in res[2]).T
        error_values = res[3].ravel()

        # save results
        _save_results(sim, process_list, param_values*x0, error_values)

        # plot
        if num_params == 1:
            plotter.plot_optimization_brute_force(res[2]*x0, res[3])


    # switch logging to the console back on
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


def _save_results(simulation: simulations.Simulations, param_names: List[Union[str, Transition]],
                  param_values: np.array, error_values: np.array,
                  full_path: str = None) -> None:
    '''Saves the results of an optimization'''
    if full_path is None:  # pragma: no cover
        full_path = simulation.save_file_full_name('optimization') + '.hdf5'
        with h5py.File(full_path, 'w') as file:
            dset = file.create_dataset("param_values", data=param_values, compression='gzip')
            dset.attrs['params'] = ', '.join(str(param) for param in param_names)
            file.create_dataset("error_values", data=error_values, compression='gzip')
            # serialze cte as text and store it as an attribute
            dset.attrs['cte'] = yaml.dump(simulation.cte)


#if __name__ == "__main__":
#    logger = logging.getLogger()
#    logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#
#    logger.debug('Called from cmd.')
#
#    import simetuc.settings as settings
#    cte = settings.load('config_file.cfg')
#
##    cte['optimization']['excitations'] = []
#
#    cte['no_console'] = False
#    cte['no_plot'] = True
#
#    optimize_dynamics(cte, average=False)

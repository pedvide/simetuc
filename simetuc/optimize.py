# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:58:58 2016

@author: Pedro
"""

import logging
import datetime
from typing import Tuple

import numpy as np
# pylint: disable=E1101
# pylint: disable=W0613
import tqdm
from lmfit import Minimizer, minimizer, Parameters, fit_report, report_fit

import simetuc.simulations as simulations
import simetuc.settings as settings
from simetuc.util import console_logger_level, EneryTransferProcess


def optim_fun(params: Parameters, sim: simulations.Simulations,
              average: bool = False) -> np.array:
    '''Update parameter values, simulate dynamics and return total error'''
    # update optimization parameter values
    for num, new_value in enumerate(params.valuesdict().values()):
        sim.cte.optimization['processes'][num].value = new_value

    # if the user didn't select several excitations to optimize, use the active one
    # otherwise, go through all requested exitations and calculate errors
    if not sim.cte.optimization.get('excitations', False):
        dynamics_sol = sim.simulate_dynamics(average=average)
        return dynamics_sol.errors
    else:
        # first switch off all excitations
        for exc_lst in sim.cte['excitations'].values():
            for excitation in exc_lst:
                excitation.active = False

        total_errors = np.zeros((sim.cte.states['activator_states'] +
                                 sim.cte.states['sensitizer_states'],), dtype=np.float64)
        # then, go through all required excitations, calculate errors and add all of them
        for exc_label in sim.cte.optimization['excitations']:
            # switch on one excitation, solve and switch off again
            sim.cte.excitations[exc_label][0].active = True
            dynamics_sol = sim.simulate_dynamics(average=average)
            sim.cte.excitations[exc_label][0].active = False
            total_errors += dynamics_sol.errors**2
        return np.sqrt(total_errors)


def optimize_dynamics(cte: settings.Settings, average: bool = False,
                      full_path: str = None) -> Tuple[np.array, float, minimizer.MinimizerResult]:
    ''' Minimize the error between experimental data and simulation for the settings in cte
        average = True -> optimize average rate equations instead of microscopic ones
    '''
    def callback_fun(params: Parameters, iter_num: int, resid: np.array,
                     sim: simulations.Simulations, average: bool = False) -> None:
        ''' This function is called after every minimization step
            It prints the current parameters and error from the cache
        '''
        optim_progbar.update(1)
        if not cte['no_console']:
            val_list = ', '.join('{:.3g}'.format(par.value) for par in params.values())
            error =  (resid*resid).sum()
            msg = '{},\t\t{:.3e},\t({})'.format(iter_num, error, val_list)
            tqdm.tqdm.write(msg)
            logger.info(msg)

    start_time = datetime.datetime.now()

    logger = logging.getLogger(__name__)
    cte['no_plot'] = True
    sim = simulations.Simulations(cte)

    method = cte.get('optimization', {}).get('method', 'leastsq').lower().replace('-', '')
    if method not in (list(minimizer.SCALAR_METHODS.keys()) +
                      list(minimizer.SCALAR_METHODS.values()) +
                      ['leastsq', 'least_squares', 'brute_force']):
        raise ValueError('Wrong optimization method ({})!'.format(method))
    logger.info('Optimization method: %s.', method)

    # Processes to optimize. If not given, all ET parameters will be optimized
    process_list = cte.optimization['processes']
    # create a set of Parameters
    params = Parameters()
    for process in process_list:
        max_val = 1e15 if isinstance(process, EneryTransferProcess) else 1
        # don't let ET processes go to zero.
        min_val = 1 if isinstance(process, EneryTransferProcess) else 0
        params.add(process.name, value=process.value, min=min_val, max=max_val)
    logger.info('Optimization parameters: ' + ', '.join(proc.name for proc in process_list) + '.')

    # log active excitations
    if not cte.optimization.get('excitations', False):
        optim_exc = []
        # look for active excitations
        for label, exc_lst in cte['excitations'].items():
            for excitation in exc_lst:
                if excitation.active is True:
                    optim_exc.append(label)
    else:
        optim_exc = cte.optimization['excitations']
    logger.info('Optimization excitations: ' + ', '.join(label for label in optim_exc) + '.')

    options_dict = {}  # type: dict
    optim_progbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
    param_names = ', '.join(proc.name for proc in process_list)
    tqdm.tqdm.write('Iter num\tError\t\tParameters ({})'.format(param_names))
    minner = Minimizer(optim_fun, params, fcn_args=(sim, average),
                       iter_cb=callback_fun)
    # minimize logging only warnings or worse to console.
    with console_logger_level(logging.WARNING):
        result = minner.minimize(method=method, **options_dict)
    optim_progbar.update(1)
    optim_progbar.close()

    report_fit(result.params)
    logger.info(fit_report(result))
#    logger.info(result.message)
    best_x = np.array([par.value for par in result.params.values()])
    if method in 'brute_force':
        min_f = result.candidates[0].score
    else:
        min_f = (result.residual**2).sum()

    total_time = datetime.datetime.now() - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print('')
    formatted_time = '{:.0f}h {:02.0f}m {:02.0f}s'.format(hours, minutes, seconds)
    logger.info('Minimum reached! Total time: %s.', formatted_time)
    logger.info('Optimized RMS error: %.3e.', min_f)
    logger.info('Parameters name and value:')
    for proc, best_val in zip(process_list, best_x.T):
        logger.info('%s: %.3e.', proc.name, best_val)

    return best_x, min_f, result


#if __name__ == "__main__":
#    logger = logging.getLogger()
#    logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#
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
#    best_x, min_f, res = optimize_dynamics(cte, average=False)

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:58:58 2016

@author: Pedro
"""

import logging
import datetime
from typing import Tuple, Callable, List
import functools

import numpy as np
# pylint: disable=E1101
# pylint: disable=W0613
import tqdm
from lmfit import Minimizer, minimizer, Parameters, fit_report, report_fit
from lmfit.minimizer import MinimizerResult

import simetuc.simulations as simulations
import simetuc.settings as settings
from simetuc.util import disable_loggers, disable_console_handler, EneryTransferProcess
from simetuc.util import save_file_full_name

class OptimSolution():
    def __init__(self, result: MinimizerResult, cte: settings.Settings, optim_progress: List[str],
                 total_time: float) -> None:
        self.result = result
        self.cte = cte
        self.optim_progress = optim_progress
        self.time = total_time

        # total time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.formatted_time = '{:.0f}h {:02.0f}m {:02.0f}s'.format(hours, minutes, seconds)

        self.best_params = np.array([par.value for par in result.params.values()])
        self.method = cte.get('optimization', {}).get('method', 'leastsq').lower().replace('-', '')
        if 'brute' in self.method:
            self.min_f = np.sqrt(result.candidates[0].score)
        else:
            self.min_f = np.sqrt((result.residual**2).sum())

    def plot(self) -> None:  # pragma: no cover
        pass

    def save(self) -> None:  # pragma: no cover
        pass

    def save_txt(self, full_path: str = None, mode: str = 'wt', cmd : str = '') -> None:
        '''Save the optimization settings to disk as a textfile'''
        logger = logging.getLogger(__name__)
        if full_path is None:  # pragma: no cover
            full_path = save_file_full_name(self.cte.lattice, 'optimization') + '.txt'
        logger.info('Saving solution as text to {}.'.format(full_path))
        # print cte
        with open(full_path, mode) as csvfile:
            csvfile.write('Settings:\n')
            csvfile.write(self.cte['config_file'])
            csvfile.write('\n\nCommand used to generate data:\n')
            csvfile.write(cmd)
            csvfile.write('\n\n\nOptimization progress:\n')
            csvfile.write(self.optim_progress[0] + '\r\n')
            for update in self.optim_progress[1:]:
                csvfile.write(update + '\r\n')

            csvfile.write('\nOptimization statistics:\n')
            csvfile.write(fit_report(self.result) + '\r\n')
            csvfile.write('\r\n')

            csvfile.write(f'Total time: {self.formatted_time}.' + '\r\n')
            csvfile.write(f'Optimized RMS error: {self.min_f:.3e}.' + '\r\n')
            csvfile.write('Parameters name and value:' + '\r\n')
            for name, best_val in zip(self.result.params.keys(), self.best_params.T):
                csvfile.write(f'{name}: {best_val:.3e}.' + '\r\n')


def optim_fun(function: Callable, params: Parameters, sim: simulations.Simulations) -> np.array:
    '''Update parameter values, simulate dynamics and return total error.
    function should be something like sim.simulate_dynamics or sim.simulate_concentration_dependence
    with no parameters, and it must return an object with an errors attribute.'''
    # update optimization parameter values
    for num, new_value in enumerate(params.valuesdict().values()):
        sim.cte.optimization['processes'][num].value = new_value

    # if the user didn't select several excitations to optimize, use the active one
    # otherwise, go through all requested exitations and calculate errors
    if not sim.cte.optimization.get('excitations', False):
        solution = function()
        return solution.errors
    else:
        # first switch off all excitations
        for exc_lst in sim.cte.excitations.values():
            for excitation in exc_lst:
                excitation.active = False

        total_errors = np.zeros((sim.cte.states['activator_states'] +
                                 sim.cte.states['sensitizer_states'],), dtype=np.float64)
        # then, go through all required excitations, calculate errors and add all of them
        for exc_label in sim.cte.optimization['excitations']:
            # switch on one excitation, solve and switch off again
            sim.cte.excitations[exc_label][0].active = True
            solution = function()
            sim.cte.excitations[exc_label][0].active = False
            total_errors += solution.errors**2
        return np.sqrt(total_errors)

def optim_fun_dynamics(params: Parameters, sim: simulations.Simulations,
                       average: bool = False, N_samples: int = None) -> np.array:
    '''Update parameter values, simulate dynamics and return error vector'''
    if N_samples is None:
        function = functools.partial(sim.simulate_dynamics, average=average)
    else:
        function = functools.partial(sim.sample_simulation, sim.simulate_dynamics, N_samples=N_samples,  # type: ignore
                                     average=average)
    return optim_fun(function, params, sim)

def optim_fun_dynamics_conc(params: Parameters, sim: simulations.Simulations,
                            average: bool = False, N_samples: int = None) -> np.array:
    '''Update parameter values, simulate dynamics for the concentrations and return error vector'''
    if N_samples is None:
        function = functools.partial(sim.simulate_concentration_dependence,
                                     sim.cte.concentration_dependence['concentrations'],
                                     sim.cte.concentration_dependence['N_uc_list'],
                                     dynamics=True, average=average)
    else:
        function = functools.partial(sim.sample_simulation, sim.simulate_concentration_dependence,  # type: ignore
                                     N_samples=N_samples,
                                     concentrations=sim.cte.concentration_dependence['concentrations'],
                                     N_uc_list=sim.cte.concentration_dependence['N_uc_list'],
                                     dynamics=True, average=average)
    return optim_fun(function, params, sim)


def setup_optim(cte: settings.Settings) -> Tuple[str, Parameters, dict]:
    '''Returns the method, process_list and options for the optimization'''
    logger = logging.getLogger(__name__)
    # multiply the values by these bounds to get the minimum and maximum allowed values
    max_factor = cte.optimization['options'].get('max_factor', 1e5)
    min_factor = cte.optimization['options'].get('min_factor', 1e-5)

    method = cte.get('optimization', {}).get('method', 'leastsq').lower().replace('-', '')
    if method not in (list(minimizer.SCALAR_METHODS.keys()) +
                      list(minimizer.SCALAR_METHODS.values()) +
                      ['leastsq', 'least_squares', 'brute']):
        raise ValueError('Wrong optimization method ({})!'.format(method))
    logger.info('Optimization method: %s.', method)

    # Processes to optimize. If not given, all ET parameters will be optimized
    process_list = cte.optimization['processes']
    # create a set of Parameters
    params = Parameters()
    for process in process_list:
        value = process.value
        max_val = max_factor*value if isinstance(process, EneryTransferProcess) else 1
        min_val = min_factor*value if isinstance(process, EneryTransferProcess) else 0
        if value == 0:
            max_val = 1e11
        params.add(process.name, value=value, min=min_val, max=max_val)
    logger.info('Optimization parameters: ' + ', '.join(proc.name for proc in process_list) + '.')

    # log active excitations
    if not cte.optimization.get('excitations', False):
        optim_exc = []
        # look for active excitations
        for label, exc_lst in cte.excitations.items():
            if exc_lst[0].active is True:
                optim_exc.append(label)
    else:
        optim_exc = cte.optimization['excitations']
    logger.info('Optimization excitations: ' + ', '.join(label for label in optim_exc) + '.')

    # optional optimization options
    options_dict = {}  # type: dict
    if 'brute' in method:
        options_dict['Ns'] = cte.optimization['options'].get('N_points', 10)

    return method, params, options_dict


def optimize(function: Callable, cte: settings.Settings, average: bool = False,
             material_text: str = '', N_samples: int = None,
             full_path: str = None) -> OptimSolution:
    ''' Minimize the error between experimental data and simulation for the settings in cte
        average = True -> optimize average rate equations instead of microscopic ones.
        function returns the error vector and accepts: parameters, sim, and average.
    '''
    logger = logging.getLogger(__name__)

    optim_progress = []  # type: List[str]

    def callback_fun(params: Parameters, iter_num: int, resid: np.array,
                     sim: simulations.Simulations,
                     average: bool = False, N_samples: int = None) -> None:
        ''' This function is called after every minimization step
            It prints the current parameters and error from the cache
        '''
        optim_progbar.update(1)
        if not cte['no_console']:
            val_list = ', '.join('{:.3e}'.format(par.value) for par in params.values())
            error =  np.sqrt((resid*resid).sum())
            msg = '{}, \t\t{}, \t{:.4e},\t[{}]'.format(iter_num,
                                                       datetime.datetime.now().strftime('%H:%M:%S'),
                                                       error, val_list)
            tqdm.tqdm.write(msg)
            logger.info(msg)
            optim_progress.append(msg)

    start_time = datetime.datetime.now()

    logger.info('Decay curves optimization of ' + material_text)

    cte['no_plot'] = True
    sim = simulations.Simulations(cte, full_path=full_path)

    method, parameters, options_dict = setup_optim(cte)

    optim_progbar = tqdm.tqdm(desc='Optimizing', unit='points', disable=cte['no_console'])
    param_names = ', '.join(name for name in parameters.keys())
    header = 'Iter num\tTime\t\tRMSD\t\tParameters ({})'.format(param_names)
    optim_progress.append(header)
    tqdm.tqdm.write(header)
    minimizer = Minimizer(function, parameters, fcn_args=(sim, average, N_samples),
                          iter_cb=callback_fun)
    # minimize logging only warnings or worse to console.
    with disable_loggers(['simetuc.simulations', 'simetuc.precalculate', 'simetuc.lattice',
                          'simetuc.simulations.conc_dep']):
        with disable_console_handler(__name__):
            result = minimizer.minimize(method=method, **options_dict)

    optim_progbar.update(1)
    optim_progbar.close()

    # fit results
    report_fit(result.params)
    logger.info(fit_report(result))

    best_x = np.array([par.value for par in result.params.values()])
    if 'brute' in method:
        min_f = np.sqrt(result.candidates[0].score)
    else:
        min_f = np.sqrt((result.residual**2).sum())

    total_time = datetime.datetime.now() - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    tqdm.tqdm.write('')
    formatted_time = '{:.0f}h {:02.0f}m {:02.0f}s'.format(hours, minutes, seconds)
    logger.info('Minimum reached! Total time: %s.', formatted_time)
    logger.info('Optimized RMS error: %.3e.', min_f)
    logger.info('Parameters name and value:')
    for name, best_val in zip(parameters.keys(), best_x.T):
        logger.info('%s: %.3e.', name, best_val)

    optim_solution = OptimSolution(result, cte, optim_progress, total_time.total_seconds())

    return optim_solution

def optimize_dynamics(cte: settings.Settings,
                      average: bool = False, full_path: str = None,
                      N_samples: int = None) -> OptimSolution:
    material = '{}: {}% {}, {}% {}.'.format(cte.lattice['name'],
                                            cte.lattice['S_conc'], cte.states['sensitizer_ion_label'],
                                            cte.lattice['A_conc'], cte.states['activator_ion_label'])
    return optimize(optim_fun_dynamics, cte, average=average, material_text=material,
                    N_samples=N_samples, full_path=full_path)

def optimize_concentrations(cte: settings.Settings,
                            average: bool = False, full_path: str = None,
                            N_samples: int = None) -> OptimSolution:
    materials = ['{}% {}, {}% {}.'.format(S_conc, cte.states['sensitizer_ion_label'],
                                          A_conc, cte.states['activator_ion_label'])
                for (S_conc, A_conc) in cte.concentration_dependence['concentrations']]
    materials_text = '{}: '.format(cte.lattice['name']) + '; '.join(materials)
    return optimize(optim_fun_dynamics_conc, cte, average=average, material_text=materials_text,
                    N_samples=N_samples, full_path=full_path)

#if __name__ == "__main__":
#    logger = logging.getLogger()
#    logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                        handlers=[logging.FileHandler("logs/optim.log"),
#                                  logging.StreamHandler()])
#
#    logger.debug('Called from cmd.')
#
#    import simetuc.settings as settings
#    cte = settings.load('config_file.cfg')

#    cte.optimization['excitations'] = []

#    cte['no_console'] = False
#    cte['no_plot'] = True

#    optim_solution = optimize_dynamics(cte, average=False, N_samples=10)


#    optim_solution = optimize_concentrations(cte)
#
#    # confidence intervals VERY SLOW
#    from lmfit import conf_interval, printfuncs
#    with disable_loggers(['simetuc.simulations', 'simetuc.precalculate', 'simetuc.lattice']):
#        with disable_console_handler(__name__):
#            ci = conf_interval(minimizer, res, sigmas=[1])
##    printfuncs.report_ci(ci, ndigits=2)
#    print(printfuncs.ci_report(ci, ndigits=1))

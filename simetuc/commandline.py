# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:53:51 2016

@author: Villanueva
"""

# TODO: INCLUDE PULSE FREQUENCY IN STEADY STATE FOR NON CW-LASER EXCITATION
# TODO: INCLUDE .CIF FILE GENERATION OF LATTICE -> doesn't work well with multiple sites...
# TODO: cooperative sensitization

import sys
import logging, logging.config
import argparse
# nice debug printing of settings
import pprint
import time
import os
import pkg_resources

import numpy as np
import matplotlib.pyplot as plt
import yaml

import simetuc.lattice as lattice
import simetuc.simulations as simulations
#from simetuc.simulations import Simulations
import simetuc.settings as settings
import simetuc.optimize as optimize


def _change_console_logger(level):
    ''' change the logging level of the console handler '''
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stdout:
                handler.setLevel(level)

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='Microscopic Rate Equation Suite')
    parser.add_argument('--version', action='version', version='Microscopic Rate Equation Suite 1.0')
    # verbose or quiet options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", help='show warnings and progress information',
                       action="store_true")
    group.add_argument("-q", "--quiet", help='show only errors', action="store_true")
    parser.add_argument("--no-plot", help='don\'t show plots', action="store_true")
    # config file
    parser.add_argument(metavar='configFilename', dest='filename', help='configuration filename')

    # main options: load config file, simulate or optimize
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', '--config', help='import configuration from file',
                        action='store_true')
    group.add_argument('-l', '--lattice', help='generate and plot the lattice',
                        action='store_true')
    group.add_argument('-d', '--dynamics', help='simulate dynamics',
                        action='store_true')
    group.add_argument('-s', '--steady-state', help='simulate steady state',
                        action='store_true')
    group.add_argument('-p', '--power-dependence', help='simulate power dependence',
                        action='store_true')
    group.add_argument('-o', '--optimize', help='optimize the energy transfer parameters',
                        action='store_true')
    args = parser.parse_args()

    # choose console logger level
    no_console = True
    if args.verbose:
        console_level = logging.INFO
        no_console = False
    elif args.quiet:
        console_level = logging.ERROR
        no_console = True
    else:
        console_level = logging.WARNING
        no_console = False

    # show plots or not
    no_plot = False
    if args.no_plot:
        no_plot = True

    # read logging settings from file
    # use the file located where the package is installed
    path = pkg_resources.get_distribution('simetuc').location
    try:
        full_path = os.path.join(path, 'simetuc', 'config', 'log_config.cfg')
        with open(full_path) as file:
            log_settings = yaml.safe_load(file)
            # modify logging to console that the user wants
            log_settings['handlers']['console']['level'] = console_level
    except OSError as err:
        print('ERROR! Logging settings file not found at {}!'.format(full_path))
        print('Logging won\'t be available!!')
        log_settings = {'version': 1} # minimum settings without errors

    # load settings and rollover any rotating file handlers
    # so each execution of this program is logged to a fresh file
    os.makedirs('logs', exist_ok=True)
    logging.config.dictConfig(log_settings)
    logger = logging.getLogger('simetuc')
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.doRollover()

    logger.debug('Called from cmd with arguments: {}.'.format(sys.argv[1:]))
    logger.debug('Log settings dump:')
    logger.debug(pprint.pformat(log_settings))

    logger.info('Starting program...')

    logger.info('Loading configuration...')
    cte = settings.load(args.filename)
    cte['no_console'] = no_console
    cte['no_plot'] = no_plot


    # choose what to do
    if args.config: # load config file
        pass

    elif args.lattice: # create lattice
        logger.info('Creating and plotting lattice...')
        lattice.generate(cte)

    elif args.steady_state: # simulate steady state
        logger.info('Simulating steady state...')
        sim = simulations.Simulations(cte)
        steady_sol = sim.simulate_steady_state()
        sim.plot_solution(steady_sol)

    elif args.power_dependence: # simulate power dependence
        logger.info('Simulating power dependence...')
        power_dens_list = np.logspace(1, 8, 8-2+1)

        # change the logging level of the console handler
        # so it only prints warnings to screen while calculating all steady states
        _change_console_logger(logging.WARNING)
        sim = simulations.Simulations(cte)
        sim.simulate_power_dependence(power_dens_list)
        # restore old level value
        _change_console_logger(console_level)
        print('')

    elif args.dynamics: # simulate dynamics
        logger.info('Simulating dynamics...')
        sim = simulations.Simulations(cte)
        dynamics_sol = sim.simulate_dynamics()
        dynamics_sol.log_errors()
        sim.plot_solution(dynamics_sol)

    if args.optimize: # optimize
        logger.info('Optimizing ET parameters...')

        _change_console_logger(logging.WARNING)

        best_x, min_f, total_time = optimize.optimize_dynamics(cte)
        print('')

        _change_console_logger(console_level)

        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Minimum reached! Total time: %s.', formatted_time)
        logger.info('Minimum value: {} at {}'.format(min_f, np.array_str(best_x, precision=5)))

    logger.info('Program finished!')

    # show all plots
    # the user needs to close the window to exit the program
    if not cte['no_plot']:
        if not args.config:
            logger.info('Close the plot window to exit.')
            plt.show()

if __name__ == "__main__":
    main()
else:
    pass
#    logger = logging.getLogger(__name__)

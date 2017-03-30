# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:22:18 2016

@author: Pedro
"""
import pytest
import numpy as np

import simetuc.optimize as optimize
import simetuc.simulations as simulations
from simetuc.settings import Settings
from simetuc.util import ExcTransition, IonType, DecayTransition, EneryTransferProcess, Transition

@pytest.fixture(scope='function')
def setup_cte():
    '''Load the cte data structure'''

    cte = {'ET': {
              'CR50': EneryTransferProcess([Transition(IonType.A, 5, 3),
                                            Transition(IonType.A, 0, 2)],
                                           mult=6, strength=2893199540.0),
              'ETU53': EneryTransferProcess([Transition(IonType.A, 5, 6),
                                             Transition(IonType.A, 3, 1)],
                                            mult=6, strength=254295690.0),
              'ETU55': EneryTransferProcess([Transition(IonType.A, 5, 6),
                                             Transition(IonType.A, 5, 4)],
                                            mult=6, strength=0.0),
              'BackET': EneryTransferProcess([Transition(IonType.A, 3, 0),
                                              Transition(IonType.S, 0, 1)],
                                             mult=6, strength=4502.0),
              'EM': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                          Transition(IonType.S, 0, 1)],
                                         mult=6, strength=45022061400.0),
              'ETU1': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                            Transition(IonType.A, 0, 2)],
                                           mult=6, strength=10000.0)
              },
         'decay': {'branching_A': [DecayTransition(IonType.A, 1, 0, branching_ratio=1.0),
                DecayTransition(IonType.A, 2, 1, branching_ratio=0.4),
                DecayTransition(IonType.A, 3, 1, branching_ratio=0.3),
                DecayTransition(IonType.A, 4, 3, branching_ratio=0.999),
                DecayTransition(IonType.A, 5, 1, branching_ratio=0.15),
                DecayTransition(IonType.A, 5, 2, branching_ratio=0.16),
                DecayTransition(IonType.A, 5, 3, branching_ratio=0.04),
                DecayTransition(IonType.A, 5, 4, branching_ratio=0.0),
                DecayTransition(IonType.A, 6, 1, branching_ratio=0.43)],
               'branching_S': [DecayTransition(IonType.S, 1, 0, branching_ratio=1.0)],
               'decay_A': [DecayTransition(IonType.A, 1, 0, decay_rate=83.33333333333333),
                DecayTransition(IonType.A, 2, 0, decay_rate=40000.0),
                DecayTransition(IonType.A, 3, 0, decay_rate=500.0),
                DecayTransition(IonType.A, 4, 0, decay_rate=500000.0),
                DecayTransition(IonType.A, 5, 0, decay_rate=1315.7894736842104),
                DecayTransition(IonType.A, 6, 0, decay_rate=14814.814814814814)],
               'decay_S': [DecayTransition(IonType.S, 1, 0, decay_rate=400.0)]},
         'excitations': {
                  'NIR_1470': [ExcTransition(IonType.A, 5, 6, False, 9/5, 2e-4, 1e7, 1e-8)],
                 'NIR_800': [ExcTransition(IonType.A, 0, 3, False, 13/9, 0.0044, 1e7, 1e-8),
                             ExcTransition(IonType.A, 2, 5, False, 11/9, 0.002, 1e7, 1e-8)],
                 'NIR_980': [ExcTransition(IonType.S, 0, 1, False, 4/3, 0.0044, 1e7, 1e-8)],
                 'Vis_473': [ExcTransition(IonType.A, 0, 5, True, 13/9, 0.00093, 1e6, 1e-8)]},
         'ions': {'activators': 113, 'sensitizers': 0, 'total': 113},
         'lattice': {'A_conc': 0.3,
          'N_uc': 20,
          'S_conc': 0.3,
          'a': 5.9738,
          'alpha': 90.0,
          'b': 5.9738,
          'beta': 90.0,
          'c': 3.5297,
          'gamma': 120.0,
          'd_max': 100.0,
          'd_max_coop': 25.0,
          'name': 'bNaYF4',
          'sites_occ': [1.0, 0.5],
          'sites_pos': [(0.0, 0.0, 0.0),
           (0.6666666666666666, 0.3333333333333333, 0.5)],
          'spacegroup': 'P-6'},
         'no_console': False,
         'no_plot': False,
         'optimization': {'method': 'SLSQP', 'processes': ['CR50']},
         'simulation_params': {'N_steps': 1000,
          'N_steps_pulse': 100,
          'atol': 1e-15,
          'rtol': 0.001},
         'states': {'activator_ion_label': 'Tm',
          'activator_states': 7,
          'activator_states_labels': ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2'],
          'energy_states': 791,
          'sensitizer_ion_label': 'Yb',
          'sensitizer_states': 2,
          'sensitizer_states_labels': ['GS', 'ES']}}

    cte['no_console'] = False
    cte['no_plot'] = True
    return Settings(cte_dict=cte)

def idfn_param(param):
    '''Returns the name of the test according to the parameters'''
    return 'method={}'.format(param)
def idfn_avg(param):
    '''Returns the name of the test according to the parameters'''
    return 'avg={}'.format(param)
@pytest.mark.parametrize('method', [None, 'COBYLA', 'L-BFGS-B', 'TNC',
                                    'SLSQP', 'brute_force', 'basin_hopping'],
                                    ids=idfn_param)
@pytest.mark.parametrize('average', [True, False], ids=idfn_avg)
def test_optim(setup_cte, mocker, method, average):
    '''Test that the optimization works'''

    # mock the simulation by returning an error that goes to 0
    mocked_opt_dyn = mocker.patch('simetuc.optimize.optim_fun_factory')
    value = 20
    def minimize(x):
        nonlocal value
        if value != 0:
            value -= 1
        return value
    mocked_opt_dyn.return_value = minimize

    setup_cte['optimization']['method'] = method
    optimize.optimize_dynamics(setup_cte, average=average)

    assert mocked_opt_dyn.called

def test_optim_no_dict_params(setup_cte, mocker):
    '''Test that the optimization works without the optimization params being present in cte'''

    # mock the simulation by returning an error that goes to 0
    mocked_opt_dyn = mocker.patch('simetuc.optimize.optim_fun_factory')
    value = 20
    def minimize(x):
        nonlocal value
        if value != 0:
            value -= 1
        return value
    mocked_opt_dyn.return_value = minimize

    del setup_cte['optimization']

    optimize.optimize_dynamics(setup_cte)

    assert mocked_opt_dyn.called


def test_optim_wrong_method(setup_cte, mocker):
    '''Test that the optimization works without the optimization params being present in cte'''

    # mock the simulation by returning an error that goes to 0
    mocked_opt_dyn = mocker.patch('simetuc.optimize.optim_fun_factory')
    value = 20
    def minimize(x):
        nonlocal value
        if value != 0:
            value -= 1
        return value
    mocked_opt_dyn.return_value = minimize

    setup_cte['optimization']['method'] = 'wrong_method'

    with pytest.raises(ValueError) as excinfo:
        optimize.optimize_dynamics(setup_cte)
    assert excinfo.match(r"Wrong optimization method!")
    assert excinfo.type == ValueError


@pytest.mark.parametrize('excitations', [[], ['Vis_473', 'NIR_980']], ids=['default_exc', 'two_exc'])
def test_optim_fun_factory(setup_cte, mocker, excitations):
    '''Test optim_fun_factory'''
    mocked_dyn = mocker.patch('simetuc.simulations.Simulations.simulate_dynamics')
    sim = simulations.Simulations(setup_cte)
    sim.cte['optimization']['excitations'] = excitations
    process_list = ['CR50', DecayTransition(IonType.A, 3, 1)]
    x0 = np.array([sim.get_ET_param_value(process) if isinstance(process, str)
                      else sim.get_branching_ratio_value(process)
                   for process in process_list])

    _update_ET_and_simulate = optimize.optim_fun_factory(sim, process_list, x0)
    _update_ET_and_simulate(x0*1.5)

    assert mocked_dyn.called


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:57:14 2017

@author: villanueva
"""

from collections import OrderedDict

import pytest

from simetuc.util import Excitation
from simetuc.util import DecayTransition, IonType, EneryTransferProcess, Transition
from simetuc.settings import Settings


@pytest.fixture(scope='function')
def setup_cte():
    '''Load the cte data structure'''

    class Cte(dict):
        __getattr__= dict.__getitem__
        __setattr__= dict.__setitem__
        __delattr__= dict.__delitem__

    cte = Cte({'version': 1,
               'energy_transfer': OrderedDict({
                  'CR50': EneryTransferProcess([Transition(IonType.A, 5, 3),
                                                Transition(IonType.A, 0, 2)],
                                               mult=6, strength=887920884.0, name='CR50'),
                  'ETU53': EneryTransferProcess([Transition(IonType.A, 5, 6),
                                                 Transition(IonType.A, 3, 1)],
                                                mult=6, strength=450220614.0, name='ETU53'),
                  'ETU55': EneryTransferProcess([Transition(IonType.A, 5, 6),
                                                 Transition(IonType.A, 5, 4)],
                                                mult=6, strength=0.0, name='ETU55'),
                  'BackET': EneryTransferProcess([Transition(IonType.A, 3, 0),
                                                  Transition(IonType.S, 0, 1)],
                                                 mult=6, strength=4502.20614, name='BackET'),
                  'EM': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                              Transition(IonType.S, 0, 1)],
                                             mult=6, strength=45022061400.0, name='EM'),
                  'ETU1': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                                Transition(IonType.A, 0, 2)],
                                               mult=6, strength=10000.0, name='ETU1'),
                  'coop1': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                             Transition(IonType.S, 1, 0),
                                             Transition(IonType.A, 0, 5)],
                                            mult=6, strength=1000.0, name='coop1')
              }),
         'decay': {
             'branching_A': {DecayTransition(IonType.A, 2, 1, branching_ratio=0.4),
                             DecayTransition(IonType.A, 3, 1, branching_ratio=0.3),
                             DecayTransition(IonType.A, 4, 3, branching_ratio=0.999),
                             DecayTransition(IonType.A, 5, 1, branching_ratio=0.15),
                             DecayTransition(IonType.A, 5, 2, branching_ratio=0.16),
                             DecayTransition(IonType.A, 5, 3, branching_ratio=0.04),
                             DecayTransition(IonType.A, 5, 4, branching_ratio=0.0),
                             DecayTransition(IonType.A, 6, 1, branching_ratio=0.43)},
             'branching_S': {DecayTransition(IonType.S, 1, 0, branching_ratio=1.0)},
             'decay_A': {DecayTransition(IonType.A, 1, 0, decay_rate=83.33333333333333),
                         DecayTransition(IonType.A, 2, 0, decay_rate=40000.0),
                         DecayTransition(IonType.A, 3, 0, decay_rate=500.0),
                         DecayTransition(IonType.A, 4, 0, decay_rate=500000.0),
                         DecayTransition(IonType.A, 5, 0, decay_rate=1315.7894736842104),
                         DecayTransition(IonType.A, 6, 0, decay_rate=14814.814814814814)},
             'decay_S': {DecayTransition(IonType.S, 1, 0, decay_rate=400.0)}
             },
         'excitations': {
                  'NIR_1470': [Excitation(IonType.A, 5, 6, False, 9/5, 2e-4, 1e7, 1e-8)],
                  'NIR_800': [Excitation(IonType.A, 0, 3, False, 13/9, 0.0044, 1e7, 1e-8),
                              Excitation(IonType.A, 2, 5, False, 11/9, 0.002, 1e7, 1e-8)],
                  'NIR_980': [Excitation(IonType.S, 0, 1, False, 4/3, 0.0044, 1e7, 1e-8)],
                  'Vis_473': [Excitation(IonType.A, 0, 5, True, 13/9, 0.00093, 1e6, 1e-8)]},
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
                     'sites_pos': [(0.0, 0.0, 0.0), (0.6666666666666666, 0.3333333333333333, 0.5)],
                     'spacegroup': 'P-6'},
         'no_console': False,
         'no_plot': False,
         'concentration_dependence': {'concentrations': [(0.0, 0.1), (0.0, 0.3), (0.0, 0.5), (0.0, 1.0)],
                                      'N_uc_list': [65, 40, 35, 25]},
         'power_dependence': [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0],
         'optimization': {'method': 'SLSQP',
                          'processes': [EneryTransferProcess([Transition(IonType.A, 5, 3), Transition(IonType.A, 0, 2)],
                                                             mult=6, strength=2893199540.0, name='CR50'),
                                        DecayTransition(IonType.A, 3, 1, branching_ratio=0.3)],
                          'options': {'tol': 1e-3,
                                      'N_points': 30,
                                      'min_factor': 1e-2,
                                      'max_factor': 2},
                          'excitations': ['Vis_473', 'NIR_980']
                          },
         'simulation_params': {'N_steps': 1000,
                               'N_steps_pulse': 2,
                               'atol': 1e-15,
                               'rtol': 0.001},
         'states': {'activator_ion_label': 'Tm',
                    'activator_states': 7,
                    'activator_states_labels': ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2'],
                    'sensitizer_ion_label': 'Yb',
                    'sensitizer_states': 2,
                    'sensitizer_states_labels': ['GS', 'ES'],
                    'energy_states': 791}
         }
         )
    cte['config_file'] = '''
version: 1

lattice:
    name: bNaYF4
    N_uc: 20
    S_conc: 0.3 # concentration
    A_conc: 0.3
    # unit cell
    a: 5.9738 # distances in Angstrom
    b: 5.9738
    c: 3.5297
    alpha: 90 # angles in degree
    beta: 90
    gamma: 120
    spacegroup: P-6 # the number is also ok for the spacegroup
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]

    # optional
    # maximum distance of interaction for normal ET and for cooperative
    # if not present, both default to infinite
    d_max: 100.0
    # it's strongly advised to keep this number low,
    # the number of coop interactions is very large (~num_atoms^3)
    d_max_coop: 25.0

states:
# all fields here are mandatory,
# leave empty if necessary (i.e.: just "sensitizer_ion_label" on a line), but don't delete them
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]

excitations:
# the excitation label can be any text
# at this point, only one active excitation is suported
# the t_pulse value is only mandatory for the dynamics, it's ignored in the steady state
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9 # initial_state_g/final_state_g
        pump_rate: 9.3e-4 # cm2/J
    NIR_1470:
        active: False
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(1G4) -> Tm(1D2) # both ion labels are required
        degeneracy: 9/5 # initial_state_g/final_state_g
        pump_rate: 2e-4 # cm2/J
    NIR_980:
        active: False
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Yb(GS)->Yb(ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J
    NIR_800: # ESA: list of processes, degeneracies and pump rates
        active: False
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: [Tm(3H6)->Tm(3H4), Tm(3H5)->Tm(1G4)]
        degeneracy: [13/9, 11/9]
        pump_rate: [4.4e-3, 2e-3] # cm2/J

sensitizer_decay:
# lifetimes in s
    ES: 2.5e-3

activator_decay:
# lifetimes in s
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6

activator_branching_ratios:
    # 3H5 and 3H4 to 3F4
    3H5->3F4: 0.4
    3H4->3F4: 0.3
    # 3F3 to 3H4
    3F3->3H4: 0.999
    # 1G4 to 3F4, 3H5, 3H4 and 3F3
    1G4->3F4: 0.15
    1G4->3H5: 0.16
    1G4->3H4: 0.04
    1G4->3F3: 0.00
    # 1D2 to 3F4
    1D2->3F4: 0.43

energy_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
        strength: 8.87920884e+08
    ETU53:
        process:  Tm(1G4) + Tm(3H4) -> Tm(1D2) + Tm(3F4)
        multipolarity: 6
        strength: 4.50220614e+08
    ETU55:
        process:  Tm(1G4) + Tm(1G4) -> Tm(1D2) + Tm(3F3)
        multipolarity: 6
        strength: 0 # 4.50220614e+7
    ETU1:
        process:  Yb(ES) + Tm(3H6) -> Yb(GS) + Tm(3H5)
        multipolarity: 6
        strength: 1e4
    BackET:
        process:  Tm(3H4) + Yb(GS) -> Tm(3H6) + Yb(ES)
        multipolarity: 6
        strength: 4502.20614
    EM:
        process:  Yb(ES) + Yb(GS) -> Yb(GS) + Yb(ES)
        multipolarity: 6
        strength: 4.50220614e+10
    coop1:
        process:  Yb(ES) + Yb(ES) + Tm(3H6) -> Yb(GS) + Yb(GS) + Tm(1G4)
        multipolarity: 6
        strength: 1000

optimization:
    processes: [CR50, 3H4->3F4]
    method: SLSQP
    options:
        tol: 1e-3
        N_points: 30
        min_factor: 1e-2
        max_factor: 2
    excitations: [Vis_473, NIR_980]

simulation_params: # default values for certain parameters in the ODE solver
    rtol: 1e-3 # relative tolerance
    atol: 1e-15 # absolute tolerance
    N_steps_pulse: 2 # number of steps for the pulse (only for dynamics)
    N_steps: 1000 # number of steps for relaxation (also for steady state)

power_dependence: [1e1, 1e7, 7]

concentration_dependence:
    concentrations: [[0], [0.1, 0.3, 0.5, 1.0]]
    N_uc_list: [65, 40, 35, 25]

'''
    cte['no_console'] = False
    cte['no_plot'] = False
    return cte



@pytest.fixture(scope='function')
def setup_cte_settings(setup_cte):
    '''Equivalent to the parsed Settings'''

    cte = setup_cte
    cte['lattice']['cell_par'] = [5.9738, 5.9738, 3.5297, 90.0, 90.0, 120.0]

    del cte['states']['energy_states']

    cte['decay']['branching_S'] = set()
    del cte['ions']
    return cte


@pytest.fixture(scope='function')
def setup_cte_sim(setup_cte):
    '''Load the settings for simulations'''

    # test_sim_dyn_2S_2A was created with these settings
    setup_cte['decay']['branching_A'].add(DecayTransition(IonType.A, 3, 2, branching_ratio=0.1))
    setup_cte['energy_transfer'] = {
               'CR50': EneryTransferProcess([Transition(IonType.A, 5, 3),
                                             Transition(IonType.A, 0, 2)],
                                            mult=6, strength=2893199540.0),
               'ETU53': EneryTransferProcess([Transition(IonType.A, 5, 6),
                                              Transition(IonType.A, 3, 1)],
                                             mult=6, strength=254295690.0),
               'BackET': EneryTransferProcess([Transition(IonType.A, 3, 0),
                                               Transition(IonType.S, 0, 1)],
                                              mult=6, strength=4502.20614),
               'EM': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                           Transition(IonType.S, 0, 1)],
                                          mult=6, strength=45022061400.0),
               'ETU1': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                             Transition(IonType.A, 0, 2)],
                                            mult=6, strength=10000.0)
               }

    return Settings.load_from_dict(setup_cte)
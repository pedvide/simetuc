# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:22:54 2016

@author: Pedro
"""
import os

import pytest
import h5py
import numpy as np

import simetuc.simulations as simulations
import simetuc.plotter as plotter
from simetuc.util import temp_config_filename, temp_bin_filename, IonType, DecayTransition
from simetuc.util import EneryTransferProcess, Transition, ExcTransition
from simetuc.settings import Settings

### TODO: Test loading exp data with different formats


test_folder_path = os.path.dirname(os.path.abspath(__file__))


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
                                             mult=6, strength=4502.20614),
              'EM': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                          Transition(IonType.S, 0, 1)],
                                         mult=6, strength=45022061400.0),
              'ETU1': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                            Transition(IonType.A, 0, 2)],
                                           mult=6, strength=10000.0),},
         'decay': {'branching_A': [DecayTransition(IonType.A, 1, 0, branching_ratio=1.0),
                DecayTransition(IonType.A, 2, 1, branching_ratio=0.4),
                DecayTransition(IonType.A, 3, 1, branching_ratio=0.3),
                DecayTransition(IonType.A, 3, 2, branching_ratio=0.1),
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
                     'sites_pos': [(0.0, 0.0, 0.0), (2/3, 1/3, 0.5)],
                     'spacegroup': 'P-6'},
         'no_console': False,
         'no_plot': False,
         'optimization': {'method': 'SLSQP', 'processes': ['CR50', 'ETU53']},
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

    cte['config_file'] = 'config_file_text'
    cte['no_console'] = True
    cte['no_plot'] = False
    return Settings(cte)

def test_sim(setup_cte):
    '''Test that the simulations work'''
    setup_cte['lattice']['S_conc'] = 0
    sim = simulations.Simulations(setup_cte)
    assert sim.cte
    assert sim

def test_sim_dyn1(setup_cte):
    '''Test that the dynamics work'''
    setup_cte['lattice']['S_conc'] = 0

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        assert sim.cte == setup_cte

        solution = sim.simulate_dynamics()
        assert solution

    solution.total_error
    solution.log_errors()
    solution.plot()
    solution.plot(state=7)
    solution.plot(state=1)

def test_sim_dyn_2S_2A(setup_cte):
    '''Test that the dynamics have the right result for a simple system'''
    test_filename = os.path.join(test_folder_path, 'data_2S_2A.hdf5')
    sim = simulations.Simulations(setup_cte, full_path=test_filename)

    solution = sim.simulate_dynamics()
    assert solution.index_A_j == [0, -1, -1, 11]
    assert solution.index_S_i == [-1, 7, 9, -1]

    with h5py.File(os.path.join(test_folder_path, 't_sol_2S_2A.hdf5')) as file:
         t_sol = np.array(file['t_sol'])
    assert np.allclose(t_sol, solution.t_sol)

    with h5py.File(os.path.join(test_folder_path, 'y_sol_2S_2A.hdf5')) as file:
         y_sol = np.array(file['y_sol'])
    assert np.allclose(y_sol, solution.y_sol)

def test_sim_dyn_wrong_state_plot(setup_cte):
    '''Test that you can't plot a wrong state.'''
    setup_cte['lattice']['S_conc'] = 0

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution = sim.simulate_dynamics()

    with pytest.raises(ValueError):
        solution.plot(state=10)

def test_sim_average_dyn(setup_cte):
    '''Test average dynamics.'''
    setup_cte['lattice']['S_conc'] = 0

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution = sim.simulate_avg_dynamics()
        assert solution

def test_sim_dyn_diff(setup_cte):
    '''Test that the two dynamics are different'''
    setup_cte['lattice']['S_conc'] = 0
    setup_cte['lattice']['A_conc'] = 0.2
    with temp_bin_filename() as temp_filename:
        sim1 = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution1 = sim1.simulate_dynamics()
        solution1.total_error

    setup_cte['ions'] = {}
    setup_cte['lattice']['S_conc'] = 0.2
    setup_cte['lattice']['A_conc'] = 0
    with temp_bin_filename() as temp_filename:
        sim2 = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution2 = sim2.simulate_dynamics()
        solution2.total_error

    assert sim1 != sim2
    assert solution1 != solution2


def test_sim_dyn_save_hdf5(setup_cte, mocker):
    '''Test that the dynamics solution is saved a loaded correctly'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution = sim.simulate_dynamics()
        assert mocked.call_count == 2

    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.DynamicsSolution.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.log_errors()
    sol_hdf5.plot()


def test_sim_dyn_save_txt(setup_cte):
    '''Test that the dynamics solution is saved a loaded correctly'''
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution = sim.simulate_dynamics()

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

def test_sim_no_file_hdf5():
    '''Wrong filename'''
    with pytest.raises(OSError):
        simulations.DynamicsSolution.load(os.path.join(test_folder_path, 'wrongFile.hdf5'))

def test_sim_dyn_no_t_pulse(setup_cte):
    '''Test that the dynamics gives an error if t_pulse is not defined'''
    del setup_cte['excitations']['Vis_473'][0].t_pulse

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        with pytest.raises(AttributeError):
            sim.simulate_dynamics()

def test_sim_steady1(setup_cte):
    '''Test that the steady state solution is saved a loaded correctly'''
    setup_cte['excitations']['Vis_473'][0].t_pulse = 1e-08
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        assert sim.cte == setup_cte
        solution = sim.simulate_steady_state()
        assert solution

    solution.log_populations()
    solution.plot()
    solution.log_populations()  # redo

    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.SteadyStateSolution.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()

def test_sim_steady2(setup_cte):
    '''Test average steady state'''
    setup_cte['excitations']['Vis_473'][0].t_pulse = 1e-08
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution = sim.simulate_avg_steady_state()
        assert solution

def test_sim_no_plot(setup_cte, recwarn):
    '''Test that no plot works'''
    setup_cte['no_plot'] = True
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution = sim.simulate_dynamics()

    with pytest.warns(plotter.PlotWarning):
        solution.plot()
#    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)

def test_sim_power_dep1(setup_cte, mocker):
    '''Test that the power dependence works'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        assert sim.cte == setup_cte
        power_dens_list = np.logspace(1, 3, 3-1+1)
        solution = sim.simulate_power_dependence(power_dens_list)
        assert mocked.call_count == len(power_dens_list)

    assert solution

    solution.plot()

    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.PowerDependenceSolution()
        assert not sol_hdf5
        sol_hdf5.load(filename)

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()

def test_sim_power_dep2(setup_cte, recwarn):
    '''Power dep list is empty'''
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        power_dens_list = []
        solution = sim.simulate_power_dependence(power_dens_list)

    with pytest.warns(plotter.PlotWarning):
        solution.plot()
#    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'Nothing to plot! The power_dependence list is emtpy!' in str(warning.message)

def test_sim_power_dep3(setup_cte, recwarn, mocker):
    '''A plot was requested, but no_plot is set'''
    setup_cte['no_plot'] = True
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.simulations.Simulations.simulate_steady_state')
        mocked.return_value = simulations.SteadyStateSolution(np.empty((1000,)),
                                                              np.empty((1000, 2*setup_cte['states']['energy_states'])),
                                                              [], [], setup_cte)

        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        power_dens_list = np.logspace(1, 2, 3)
        solution = sim.simulate_power_dependence(power_dens_list)
        assert mocked.call_count == len(power_dens_list)

    with pytest.warns(plotter.PlotWarning):
        solution.plot()
#    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)

def test_sim_power_dep4(setup_cte, mocker):
    '''Check that the solutions have the right power_dens'''
    setup_cte['no_plot'] = True
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        power_dens_list = np.logspace(1, 3, 3)
        solution = sim.simulate_power_dependence(power_dens_list)
        assert mocked.call_count == len(power_dens_list)

    for num, pow_dens in enumerate(power_dens_list):
        assert solution[num].power_dens == pow_dens

def test_sim_conc_dep_steady(setup_cte, mocker):
    '''Test that the concentration dependence works'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
        assert mocked.call_count == len(conc_list)

    assert solution
    solution.plot()
    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.ConcentrationDependenceSolution()
        assert not sol_hdf5
        sol_hdf5.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()

def test_sim_conc_dep_dyn(setup_cte, mocker):
    '''Test that the concentration dependence works'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=True)
        # dynamics call solve_ode twice (pulse and relaxation)
        assert mocked.call_count == 2*len(conc_list)

    assert solution
    solution.plot()
    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.ConcentrationDependenceSolution()
        assert not sol_hdf5
        sol_hdf5.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()

def test_sim_conc_dep_list(setup_cte, mocker):
    '''Test that the concentration dependence works'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=True)
        # dynamics call solve_ode twice (pulse and relaxation)
        assert mocked.call_count == 2*len(conc_list)

        for num, conc in enumerate(conc_list):
            assert solution[num].concentration == conc

def test_sim_conc_dep_only_A(setup_cte, mocker):
    '''Conc list has only A changing'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        conc_list = [(0.0, 0.1), (0.0, 0.2), (0.0, 0.3)]
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
        assert mocked.call_count == len(conc_list)

    assert solution
    solution.plot()

def test_sim_conc_dep_only_S(setup_cte, mocker):
    '''Conc list has only S changing'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        conc_list = [(0.01, 0.3), (0.1, 0.3), (0.3, 0.3)]
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
        assert mocked.call_count == len(conc_list)

    assert solution
    solution.plot()

def test_sim_conc_dep_empty_conc(setup_cte, recwarn):
    '''Conc list is empty'''
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        conc_list = []
        solution = sim.simulate_concentration_dependence(conc_list)

    with pytest.warns(plotter.PlotWarning):
        solution.plot()
#    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'Nothing to plot! The concentration_dependence list is emtpy!' in str(warning.message)


def test_sim_conc_dep_no_plot(setup_cte, recwarn, mocker):
    '''A plot was requested, but no_plot is set'''
    setup_cte['no_plot'] = True
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte['states']['energy_states']))

        sim = simulations.Simulations(setup_cte, full_path=temp_filename)
        conc_list = [(0.01, 0.3), (0.1, 0.3)]
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
        assert mocked.call_count == len(conc_list)

    with pytest.warns(plotter.PlotWarning):
        solution.plot()
#    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)


def test_sim_conc_dep_no_file():
    '''Wrong filename'''
    with pytest.raises(OSError):
        simulations.PowerDependenceSolution().load(os.path.join(test_folder_path, 'wrongFile.hdf5'))


# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:22:18 2016

@author: Pedro
"""
import pytest
import numpy as np
import warnings

from lmfit import Parameters

import simetuc.optimize as optimize
import simetuc.simulations as simulations
from simetuc.util import IonType, DecayTransition, EneryTransferProcess, Transition
from simetuc.util import temp_bin_filename, temp_config_filename


B_43 = DecayTransition(IonType.A, 3, 1, branching_ratio=0.3)
CR50 = EneryTransferProcess([Transition(IonType.A, 5, 3), Transition(IonType.A, 0, 2)],
                            mult=6, strength=2893199540.0, name='CR50')
def idfn_param(param):
    '''Returns the name of the test according to the parameters'''
    return 'method={}'.format(param)
def idfn_avg(param):
    '''Returns the name of the test according to the parameters'''
    return 'avg={}'.format(param)
def idfn_proc(param):
    '''Returns the name of the test according to the parameters'''
    return 'num={}'.format(len(param))
@pytest.mark.parametrize('method', ['COBYLA', 'L-BFGS-B', 'TNC',
                                    'SLSQP', 'brute_force', 'leastsq'],
                                    ids=idfn_param)
@pytest.mark.parametrize('function', ['optimize_dynamics', 'optimize_concentrations'])
@pytest.mark.parametrize('average', [True, False], ids=idfn_avg)
@pytest.mark.parametrize('processes', [[CR50, B_43],
                                       [CR50],
                                       [B_43]], ids=idfn_proc)
@pytest.mark.parametrize('excitations', [[], ['Vis_473'],  ['Vis_473', 'NIR_980']],
                         ids=['default_exc', 'one_exc', 'two_exc'])
def test_optim(setup_cte, mocker, method, function, average, processes, excitations):
    '''Test that the optimization works'''
    # mock the simulation by returning an error that goes to 0
    init_param = np.array([proc.value for proc in processes])
    def mocked_optim_fun(function, params, sim):
        return 2 + (np.array([val for val in params.valuesdict().values()]) - 1.1*init_param)**2
    mocker.patch('simetuc.optimize.optim_fun', new=mocked_optim_fun)

    setup_cte['optimization']['method'] = method
    setup_cte['optimization']['processes'] = processes
    setup_cte['optimization']['excitations'] = excitations
    fun = getattr(optimize, function)
    with warnings.catch_warnings(), temp_bin_filename() as temp_filename:
        warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
        optim_solution = fun(setup_cte, average=average, full_path=temp_filename)
        best_x = optim_solution.best_params
        min_f = optim_solution.min_f
        res = optim_solution.result

    assert len(best_x) == len(processes)
    if method in 'brute_force':
        assert min_f == np.sqrt(res.candidates[0].score)
    else:
        assert min_f == np.sqrt((res.residual**2).sum())

def test_optim_no_dict_params(setup_cte, mocker):
    '''Test that the optimization works with an empty optimization dict'''
    # mock the simulation by returning an error that goes to 0
    init_param = np.array([proc.value for proc in setup_cte.energy_transfer.values() if proc.value != 0])
    def mocked_optim_fun(function, params, sim):
        return 2 + (np.array([val for val in params.valuesdict().values()]) - 1.1*init_param)**2
    mocker.patch('simetuc.optimize.optim_fun', new=mocked_optim_fun)

    setup_cte['optimization'] = {}
    setup_cte['optimization']['processes'] = [proc for proc in setup_cte.energy_transfer.values() if proc.value != 0]
    setup_cte['optimization']['options'] = {}
    with temp_bin_filename() as temp_filename:
        optim_solution = optimize.optimize_dynamics(setup_cte, full_path=temp_filename)
        best_x = optim_solution.best_params
        min_f = optim_solution.min_f
        res = optim_solution.result

    assert len(best_x) == len(init_param)
    assert min_f == np.sqrt((res.residual**2).sum())


def test_optim_wrong_method(setup_cte, mocker):
    '''Test that the optimization works without the optimization params being present in cte'''
    # mock the simulation by returning an error that goes to 0
    init_param = np.array([proc.value for proc in setup_cte['optimization']['processes']])
    def mocked_optim_fun(function, params, sim):
        return 2 + (np.array([val for val in params.valuesdict().values()]) - 1.1*init_param)**2
    mocker.patch('simetuc.optimize.optim_fun', new=mocked_optim_fun)

    setup_cte['optimization']['method'] = 'wrong_method'

    with pytest.raises(ValueError) as excinfo:
        with temp_bin_filename() as temp_filename:
            optimize.optimize_dynamics(setup_cte, full_path=temp_filename)
    assert excinfo.match(r"Wrong optimization method")
    assert excinfo.type == ValueError


@pytest.mark.parametrize('excitations', [[], ['Vis_473'],  ['Vis_473', 'NIR_980']],
                         ids=['default_exc', 'one_exc', 'two_exc'])
def test_optim_fun(setup_cte, mocker, excitations):
    '''Test optim_fun'''
    mocked_dyn = mocker.patch('simetuc.simulations.Simulations.simulate_dynamics')
    class mocked_dyn_res:
        errors = np.ones((setup_cte.states['activator_states'] +
                          setup_cte.states['sensitizer_states'],), dtype=np.float64)
        average = False
    mocked_dyn.return_value = mocked_dyn_res

    sim = simulations.Simulations(setup_cte)
    sim.cte['optimization']['excitations'] = excitations

    # Processes to optimize. If not given, all ET parameters will be optimized
    process_list = setup_cte.optimization['processes']
    # create a set of Parameters
    params = Parameters()
    for process in process_list:
        max_val = 1e15 if isinstance(process, EneryTransferProcess) else 1
        # don't let ET processes go to zero.
        min_val = 1 if isinstance(process, EneryTransferProcess) else 0
        params.add(process.name, value=process.value, min=min_val, max=max_val)

    optimize.optim_fun_dynamics(params, sim, average=False)

    sim.cte['concentration_dependence']['concentrations'] = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
    optimize.optim_fun_dynamics_conc(params, sim)

    assert mocked_dyn.called

def test_optim_save_txt(setup_cte, mocker):
    '''Test that the optim solution is saved as text correctly'''
    init_param = np.array([proc.value for proc in setup_cte['optimization']['processes']])
    def mocked_optim_fun(function, params, sim):
        return 2 + (np.array([val for val in params.valuesdict().values()]) - 1.1*init_param)**2
    mocker.patch('simetuc.optimize.optim_fun', new=mocked_optim_fun)

    with temp_bin_filename() as temp_filename:
        solution = optimize.optimize_dynamics(setup_cte, full_path=temp_filename)

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:48:45 2017

@author: villanueva
"""
from fractions import Fraction
import sys
from typing import List, Tuple, Union, Dict
from simetuc.value import Value, DictValue

class f_float(type):
    '''Type that converts numbers or str into floats through Fraction.'''
    def __new__(mcs, x: str) -> float:
        '''Return the float'''
        return float(Fraction(x))  # type: ignore
f_float.__name__ = 'float(Fraction())'

# smallest float number
min_float = sys.float_info.min

Vector = Tuple[f_float]
settings = [DictValue('lattice', [Value('name', str),
                                  Value('spacegroup', Union[int, str]),
                                  Value('S_conc', float, val_min=0, val_max=100),
                                  Value('A_conc', float, val_min=0, val_max=100),
                                  Value('a', float, val_min=0),
                                  Value('b', float, val_min=0),
                                  Value('c', float, val_min=0),
                                  Value('alpha', float, val_min=0, val_max=360),
                                  Value('beta', float, val_min=0, val_max=360),
                                  Value('gamma', float, val_min=0, val_max=360),
                                  Value('sites_pos', Union[Vector, List[Vector]],
                                        val_min=0, val_max=1,
                                        len_min=[1, 3], len_max=[None, 3]),
                                  Value('sites_occ', Union[f_float, List[f_float]],
                                        val_min=0, val_max=1, len_min=1),
                                  Value('d_max', float, val_min=0, kind=Value.optional),
                                  Value('d_max_coop', float, val_min=0, kind=Value.optional),
                                  Value('N_uc', int, val_min=1, kind=Value.exclusive),
                                  Value('radius', float, val_min=min_float, kind=Value.exclusive)]),

            DictValue('states', [Value('sensitizer_ion_label', str),
                                 Value('activator_ion_label', str),
                                 Value('sensitizer_states_labels', List[str], len_min=1),
                                 Value('activator_states_labels', List[str], len_min=1)]),

            Value('excitations',
                  Dict[str, DictValue('', [Value('active', bool),  # type: ignore
                                           Value('power_dens', float, val_min=0),
                                           Value('process', Union[List[str], str], len_min=1),
                                           Value('degeneracy', Union[f_float, List[f_float]],
                                                 val_min=0, len_min=1),
                                           Value('pump_rate', Union[float, List[float]],
                                                 val_min=0, len_min=1),
                                           Value('t_pulse', float, kind=Value.optional)])
                      ]),

            Value('sensitizer_decay', Dict[str, float]),
            Value('activator_decay', Dict[str, float]),

            Value('sensitizer_branching_ratios', Dict[str, float], val_min=0,
                  val_max=1, kind=Value.optional),
            Value('activator_branching_ratios', Dict[str, float], val_min=0,
                  val_max=1, kind=Value.optional),

            Value('energy_transfer', Dict[str, DictValue('',  # type: ignore
                                                         [Value('process', str),
                                                          Value('multipolarity', float, val_min=0),
                                                          Value('strength', float, val_min=0),
                                                          Value('strength_avg', float, val_min=0,
                                                                kind=Value.optional)])],
                  kind=Value.optional),

            DictValue('optimization', [Value('processes', List[str], kind=Value.optional),
                                       Value('method', str, kind=Value.optional),
                                       Value('excitations', List[str], kind=Value.optional),
                                       DictValue('options', [Value('N_points', int, kind=Value.optional),
                                                             Value('max_factor', float, kind=Value.optional),
                                                             Value('min_factor', float, kind=Value.optional),
                                                             Value('tol', float, kind=Value.optional)],
                                    	 kind=Value.optional)],
                      kind=Value.optional),

            DictValue('simulation_params', [Value('rtol', float, kind=Value.optional),
                                            Value('atol', float, kind=Value.optional),
                                            Value('N_steps_pulse', int, kind=Value.optional),
                                            Value('N_steps', int, kind=Value.optional),
                                           ],
                    	 kind=Value.optional),

            Value('power_dependence', List[float], len_min=3, len_max=3, kind=Value.optional),

            Value('concentration_dependence', List[List[float]], val_min=0, val_max=100,
                  len_min=[2, None], len_max=[2, None], kind=Value.optional),
           ]

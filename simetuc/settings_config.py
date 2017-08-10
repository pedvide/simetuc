# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:48:45 2017

@author: villanueva
"""
from fractions import Fraction
import sys
from typing import List, Tuple, Union, Dict

from settings_parser import Value, DictValue, Kind

class f_float(type):
    '''Type that converts numbers or str into floats through Fraction.'''
    def __new__(mcs, x: str) -> float:
        '''Return the float'''
        return float(Fraction(x))
f_float.__name__ = 'float(Fraction())'

# smallest float number
min_float = sys.float_info.min

Vector = Tuple[f_float, f_float, f_float]

settings = {'version': Value(int, val_max=1, val_min=1),
            'lattice': DictValue({'name': str,
                                  'spacegroup': Value(Union[int, str]),
                                  'S_conc' : Value(float, val_min=0, val_max=100),
                                  'A_conc' : Value(float, val_min=0, val_max=100),
                                  'a' : Value(float, val_min=0),
                                  'b' : Value(float, val_min=0),
                                  'c' : Value(float, val_min=0),
                                  'alpha' : Value(float, val_min=0, val_max=360),
                                  'beta' : Value(float, val_min=0, val_max=360),
                                  'gamma' : Value(float, val_min=0, val_max=360),
                                  'sites_pos' : Value(Union[Vector, List[Vector]],
                                                      val_min=0, val_max=1,
                                                      len_min=[1, 3], len_max=[None, 3]),
                                  'sites_occ' : Value(Union[f_float, List[f_float]],
                                                      val_min=0, val_max=1, len_min=1),
                                  'd_max' : Value(float, val_min=0, kind=Kind.optional),
                                  'd_max_coop' : Value(float, val_min=0, kind=Kind.optional),
                                  'N_uc' : Value(int, val_min=1, kind=Value.exclusive),
                                  'radius' : Value(float, val_min=min_float, kind=Value.exclusive),
                                  }),
            'states': DictValue({'sensitizer_ion_label': str,
                                 'activator_ion_label': str,
                                 'sensitizer_states_labels' : Value(List[str], len_min=1),
                                 'activator_states_labels' : Value(List[str], len_min=1),
                                 }),
            'excitations': Value(Dict[str, DictValue({'active': bool,  # type: ignore
                                                      'power_dens': Value(float, val_min=0),
                                                      'process' : Value(Union[List[str], str], len_min=1),
                                                      'degeneracy': Value(Union[f_float, List[f_float]],
                                                                          val_min=0, len_min=1),
                                                      'pump_rate': Value(Union[float, List[float]],
                                                                         val_min=0, len_min=1),
                                                      't_pulse': Value(float, kind=Value.optional)
                                                    })
                                     ]),
            'sensitizer_decay': Value(Dict[str, float]),
            'activator_decay': Value(Dict[str, float]),

            'sensitizer_branching_ratios': Value(Dict[str, float], val_min=0, val_max=1,
                                                 kind=Value.optional),
            'activator_branching_ratios': Value(Dict[str, float], val_min=0, val_max=1,
                                                kind=Value.optional),

            'energy_transfer': Value(Dict[str, DictValue({'process': str,  # type: ignore
                                                          'multipolarity': Value(float, val_min=0),
                                                          'strength': Value(float, val_min=0),
                                                          'strength_avg': Value(float, val_min=0,
                                                                                kind=Value.optional),
                                                         })
                                         ], kind=Value.optional),

            'optimization': DictValue({'processes': Value(List[str], kind=Value.optional),
                                       'method': Value(str, kind=Value.optional),
                                       'excitations': Value(List[str], kind=Value.optional),
                                       'options': DictValue({'N_points': Value(int, kind=Value.optional),
                                                             'max_factor': Value(float, kind=Value.optional),
                                                             'min_factor': Value(float, kind=Value.optional),
                                                             'tol': Value(float, kind=Value.optional)},
                                                         	 kind=Value.optional)},
                                      kind=Value.optional),

            'simulation_params': DictValue({'rtol': Value(float, kind=Value.optional),
                                            'atol': Value(float, kind=Value.optional),
                                            'N_steps_pulse': Value(int, kind=Value.optional),
                                            'N_steps': Value(int, kind=Value.optional),
                                           }, kind=Value.optional),

            'power_dependence': Value(List[float], len_min=3, len_max=3, kind=Value.optional),

            'concentration_dependence': DictValue({'concentrations': Value(List[List[float]], val_min=0, val_max=100,
                                                                           len_min=[2, None], len_max=[2, None]),
                                                   'N_uc_list': Value(List[int], val_min=0,  kind=Value.optional)
                                                   }, kind=Value.optional),

           }

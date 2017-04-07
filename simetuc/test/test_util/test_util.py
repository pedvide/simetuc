# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:03:04 2017

@author: Villanueva
"""

import pytest

from simetuc.util import IonType, Transition, DecayTransition, Excitation, EneryTransferProcess

def test_transition():
    '''Test generic transitions'''

    # labels are skipped in comparisons and hashing
    t1 = Transition(IonType.S, 1, 0)
    t2 = Transition(IonType.S, 1, 0, label_ion='Yb', label_i='ES', label_f='GS')
    assert t1 == t2
    assert hash(t1) == hash(t2)

    # different states yield different Transitions
    t3 = Transition(IonType.S, 2, 0)
    assert t3 != t1
    assert t3 != t2
    assert hash(t3) != hash(t1)
    assert hash(t3) != hash(t2)

    # different ions yield different Transitions
    t4 = Transition(IonType.A, 1, 0)
    assert t4 != t1
    assert hash(t4) != hash(t1)


def test_decay_transition():
    '''Test generic decay or branching transitions'''

    # labels are skipped in comparisons and hashing
    t1 = DecayTransition(IonType.S, 1, 0, branching_ratio=0.4)
    t2 = DecayTransition(IonType.S, 1, 0, branching_ratio=0.4,
                         label_ion='Yb', label_i='ES', label_f='GS')
    assert t1 == t2
    assert hash(t1) == hash(t2)
    t1_dec = DecayTransition(IonType.S, 1, 0, decay_rate=1e5)
    t2_dec = DecayTransition(IonType.S, 1, 0, decay_rate=1e5,
                             label_ion='Yb', label_i='ES', label_f='GS')
    assert t1_dec == t2_dec
    assert hash(t1_dec) == hash(t2_dec)

    # different branching_ratios: not equal but same hash
    t3 = DecayTransition(IonType.S, 1, 0, branching_ratio=0.1)
    assert t3 != t1
    assert t3 != t2
    assert hash(t3) == hash(t1)
    assert hash(t3) == hash(t2)

    # compare Transitions with DecayTransitions, ignores values
    t4 = Transition(IonType.S, 1, 0)
    assert t1 == t4
    assert t4 == t1
    assert hash(t1) == hash(t4)


def test_exc_transition():
    '''Test excitation transitions'''

    t1 = Excitation(IonType.S, 1, 0, active=True, degeneracy=4/3, pump_rate=4e-2,
                    power_dens=1e7, t_pulse=1e-8)
    t2 = Excitation(IonType.S, 1, 0, active=True, degeneracy=4/3, pump_rate=4e-2,
                    power_dens=1e7, t_pulse=1e-8,
                    label_ion='Yb', label_i='ES', label_f='GS')
    assert t1 == t2

    # different params
    t3 = Excitation(IonType.S, 1, 0, active=False, degeneracy=4/3, pump_rate=4e-2,
                    power_dens=1e7, t_pulse=1e-8)
    assert t3 != t1
    assert t3 != t2


def test_ET_process():
    '''Test excitation transitions'''

    strength = 1e9
    t1 = Transition(IonType.S, 1, 0)
    t2 = Transition(IonType.S, 0, 1)
    et1 = EneryTransferProcess([t1, t2], mult=6, strength=strength)
    et2 = EneryTransferProcess([t1, t2], mult=8, strength=strength)
    assert et1 != et2

    t3 = Transition(IonType.S, 0, 1, label_ion='Yb')
    et3 = EneryTransferProcess([t1, t3], mult=6, strength=strength)
    assert et1 == et3

    assert et1.strength == strength
    assert et1.strength_avg == strength

    strength_avg = 1e3
    et4 = EneryTransferProcess([t1, t2], mult=8, strength=strength, strength_avg=strength_avg)
    assert et4.strength_avg == strength_avg



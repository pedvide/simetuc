# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:45:53 2017

@author: Villanueva
"""

import pytest
#import numpy as np
from fractions import Fraction

from simetuc.settings import Value
from typing import Dict, List, Tuple, Set, Union



def idfn(value):
    '''Returns the name of the test according to the parameters'''
    return str(type(value).__name__) + '_' + str(value)


@pytest.mark.parametrize('value', [5, 1.25, 1+5j, Fraction(2,5),
                                   'a', 'asd', '\u00B5', {'a': 2},
                                   True, b'2458', [1,2,3], (4,5,6), {7,8,9}], ids=idfn)
def test_always_right_casts(value):
    '''Parsing a value of the same type as expected should aways work
        and return the same value.'''
    assert Value('val', type(value)).parse(value) == value


def test_wrong_casts():
    '''Parse values that cannot be converted to the requested type.'''
    with pytest.raises(ValueError) as excinfo:
        Value('val', int).parse('50.0')
    assert excinfo.match("does not have the right type")
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', int).parse(5+0j)
    assert excinfo.match("does not have the right type")
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', bytes).parse('asd')
    assert excinfo.match("does not have the right type")
    assert excinfo.type == ValueError


def test_str_to_num():
    '''Test conversion from str to numbers.'''

    assert Value('val', int).parse('25') == 25
    assert Value('val', float).parse('25.0') == 25.0
    assert Value('val', complex).parse('2+5j') == 2+5j
    assert Value('val', Fraction).parse('79/98') == Fraction(79,98)


def test_num_to_str():
    '''Test conversion from numbers to strings.'''

    assert Value('val', str).parse(25) == '25'
    assert Value('val', str).parse(25.0) == '25.0'
    assert Value('val', str).parse(2+5j) == '(2+5j)'
    assert Value('val', str).parse(Fraction(78/98)) == '3584497662601007/4503599627370496'


def test_num_to_num():
    '''Test converions between numbers of different types.'''

    # int, Fraction to float
    assert Value('val', float).parse(25) == 25.0
    assert Value('val', float).parse(Fraction(89,5)) == 17.8
    # float, Fraction to int
    assert Value('val', int).parse(12.2) == 12
    assert Value('val', int).parse(Fraction(10,5)) == 2
    # int, float to Fraction
    assert Value('val', Fraction).parse(7) == Fraction(7,1)
    assert Value('val', Fraction).parse(56.2) == Fraction(3954723422784717, 70368744177664)
    # int, float, Fraction to complex
    assert Value('val', complex).parse(25) == 25+0j
    assert Value('val', complex).parse(12.2) == 12.2+0j
    assert Value('val', complex).parse(Fraction(10,5)) == 2+0j


def test_max_min_val():
    '''Test that max and min values work'''

    assert Value('val', int, val_max=30, val_min=5).parse(25) == 25
    assert Value('val', int, val_max=30, val_min=5).parse(30) == 30
    assert Value('val', int, val_max=30, val_min=5).parse(5) == 5

    with pytest.raises(ValueError) as excinfo:
        Value('val', int, val_max=30, val_min=5).parse(50)
    assert excinfo.match("cannot be larger than")
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', int, val_max=30, val_min=5).parse(-5)
    assert excinfo.match("cannot be smaller than")
    assert excinfo.type == ValueError


def test_simple_unions():
    '''Test that Unions of simple types go through each member.'''
    assert Value('val', Union[int, str]).parse(12.0) == 12
    assert Value('val', Union[int, str]).parse('12.0') == '12.0'

    assert Value('val', Union[float, complex]).parse(5+0j) == 5+0j
    assert Value('val', Union[float, Fraction]).parse('5/2') == Fraction(5,2)

    with pytest.raises(ValueError) as excinfo:
        Value('val', Union[int, float]).parse(5+1j)
    assert excinfo.match("does not have any of the right types")
    assert excinfo.type == ValueError


def test_nested_unions():
    '''Test Unions of Sequences and simple types'''
    assert Value('val', Union[int, List[int]]).parse(12) == 12
    assert Value('val', Union[int, List[int]]).parse([12]) == [12]
    assert Value('val', Union[int, List[int]]).parse([12, 13, 14]) == [12, 13, 14]

    assert Value('val', Union[List[int],
                              List[List[int]]]).parse([[1,2,3],
                                                       [4,5,6]]) ==  [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(ValueError) as excinfo:
        Value('val', Union[int, List[int]]).parse(5+1j)
    assert excinfo.match("does not have any of the right types")
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', Union[int, List[int]], len_max=2).parse([12, 13, 14])
    assert excinfo.match("does not have any of the right types")
    assert excinfo.type == ValueError


def test_own_types():
    '''Test that user-defined types, such as combination of other types, work.'''
    class f_float:
        '''Simulates a type that converts numbers or str into floats through Fraction.'''
        def __new__(cls, x: str) -> float:
            '''Return the float'''
            return float(Fraction(x))  # type: ignore
    f_float.__name__ = 'float(Fraction())'

    # good
    assert Value('val', f_float).parse('4/5') == 0.8

    # wrong
    with pytest.raises(ValueError) as excinfo:
        Value('val', f_float).parse('asd')
    assert excinfo.match(r"does not have the right type \(float\(Fraction\(\)\)\).")
    assert excinfo.type == ValueError


def test_simple_lists():
    '''Test lists of simple types.'''
    assert Value('val', List[int]).parse([1, 2]) == [1, 2]
    assert Value('val', List[float]).parse([1, 2.5, -9.1]) == [1, 2.5, -9.1]
    assert Value('val', List[str]).parse([1, 'a', 'asd', '\u2569']) == ['1', 'a', 'asd', '╩']
    assert Value('val', List[Union[int, str]]).parse([5, '6.0', 'a']) == [5, '6.0', 'a']

    with pytest.raises(ValueError) as excinfo:
        Value('val', List[int]).parse('56')
    assert excinfo.match('does not have the right type')
    assert excinfo.type == ValueError


def test_simple_list_len():
    '''Test the max and min length of a simple list.'''
    assert Value('val', List[int], len_max=4, len_min=2).parse([1, 2, 3]) == [1, 2, 3]
    assert Value('val', List[int], len_max=4, len_min=1).parse([1,2]) == [1, 2]
    assert Value('val', List[int], len_max=4, len_min=2).parse([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert Value('val', List[int], len_max=3, len_min=3).parse([1, 2, 3]) == [1, 2, 3]
    assert Value('val', List[int], len_min=3).parse([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert Value('val', List[int], len_max=3).parse([1, 2]) == [1, 2]

    with pytest.raises(ValueError) as excinfo:
        Value('val', List[int], len_max=4, len_min=2).parse([1])
    assert excinfo.match('cannot be smaller than')
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', List[int], len_max=4, len_min=2).parse([1, 2, 3, 4, 5])
    assert excinfo.match('cannot be larger than')
    assert excinfo.type == ValueError


def test_list_len_val():
    '''Test both the list's length and the size of the values'''
    assert Value('val', List[int], val_max=5, val_min=1,
                 len_max=4, len_min=1).parse([1,2]) == [1, 2]
    assert Value('val', List[int], val_max=5, val_min=1,
                 len_max=4, len_min=1).parse([4]) == [4]
    assert Value('val', List[int], val_max=5, val_min=1,
                 len_max=4, len_min=1).parse([1,2]) == [1, 2]


def test_simple_sequences():
    '''Test tuples, sets'''
    assert Value('val', Tuple[int]).parse([1, 2]) == (1, 2)
    assert Value('val', Set[float]).parse([1, 2.5, -9.1, 1]) == {1, 2.5, -9.1}
#    assert Value('val', Deque[str]).parse('hjkl') == deque(['h', 'j', 'k', 'l'])


def test_nested_lists():
    '''Test lists of lists of simple types'''
    assert Value('val', List[List[int]]).parse([[1, 2, 3], [4, 5, 6]]) == [[1, 2, 3], [4, 5, 6]]
    assert Value('val', List[List[str]]).parse([['asd', 'dsa'],
                                                ['t', 'y', 'i']]) == [['asd', 'dsa'],
                                                                      ['t', 'y', 'i']]
    assert Value('val', List[Set[Tuple[int]]]).parse([[[1,2],[1,2]],
                                                     [[4,5], [6,7]]]) == [{(1, 2)}, {(4, 5),
                                                                                     (6, 7)}]

    # this is a list of lists of str, and works
    assert Value('val', List[List[str]]).parse([['a'], ['s'], ['d'],
                                                ['d'], ['s'], ['a']]) == [['a'], ['s'], ['d'],
                                                                          ['d'], ['s'], ['a']]
    assert Value('val', List[List[List[str]]]).parse([[[1], [2]],
                                                      [[4], [5]]]) == [[['1'], ['2']],
                                                                       [['4'], ['5']]]


    with pytest.raises(ValueError) as excinfo:
        # this is a list of str, not lists of list of str (even though str is iterable)!!
        Value('val', List[List[str]]).parse(['asddsa'])
    assert excinfo.match('does not have the right type')
    assert excinfo.type == ValueError


def test_nested_lists_len():
    '''Test the length of nested lists'''
    assert Value('val', List[List[int]]).parse([[1, 2], [4, 5]]) == [[1, 2], [4, 5]]

    assert Value('val', List[List[int]],
                 len_max=[2,2]).parse([[1, 2], [4, 5]]) == [[1, 2], [4, 5]]
    assert Value('val', List[List[int]],
                 len_min=[2,3]).parse([[1, 2, 3], [4, 5, 6]]) == [[1, 2, 3], [4, 5, 6]]
    assert Value('val', List[List[int]],
                 len_max=[None,2]).parse([[1, 2], [4, 5], [4, 5]]) == [[1, 2], [4, 5], [4, 5]]


    with pytest.raises(ValueError) as excinfo:
        Value('val', List[List[int]], len_max=[2,2]).parse([[1, 2], [4, 5], [7, 8]])
    assert excinfo.match('cannot be larger than')
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', List[List[int]], len_min=[2,4]).parse([[1, 2, 3], [4, 5, 6, 7]])
    assert excinfo.match('cannot be smaller than')
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', List[List[int]], len_max=[None,2]).parse([[1, 2], [4, 5], [4, 5, 5]])
    assert excinfo.match('cannot be larger than')
    assert excinfo.type == ValueError


def test_simple_dicts():
    '''Test the parsing of simple dictionaries.'''
    d = {'a': '1'}
    assert Value('val', Dict[str, str]).parse(d) == d
    assert Value('val', Dict[str, int]).parse(d) == {'a': 1}
    assert Value('val', Dict[str, float]).parse(d) == {'a': 1.0}
    assert Value('val', Dict[str, complex]).parse(d) == {'a': (1+0j)}

    d2 = {1: 5}
    assert Value('val', Dict[str, str]).parse(d2) == {'1': '5'}
    assert Value('val', Dict[str, int]).parse(d2) == {'1': 5}
    assert Value('val', Dict[int, str]).parse(d2) == {1: '5'}
    assert Value('val', Dict[int, int]).parse(d2) == {1: 5}

    assert Value('val', Dict[str, Union[int, str]]).parse({'a': 1}) == {'a': 1}
    assert Value('val', Dict[str, Union[int, str]]).parse({'a': '1'}) == {'a': 1}
    assert Value('val', Dict[str, Union[int, str]]).parse({'a': 'b'}) == {'a': 'b'}

    assert Value('val', Dict[str, List[int]]).parse({'a': [1, 2, 3]}) == {'a': [1, 2, 3]}
    assert Value('val', Dict[str, List[int]],
                 len_max=[1,3]).parse({'a': [1, 2, 3]}) == {'a': [1, 2, 3]}

    assert Value('val', Dict[str, Set[int]],
                 len_max=[1,3]).parse({'a': [1, 2, 2]}) == {'a': {1, 2}}

    with pytest.raises(ValueError) as excinfo:
        Value('val', Dict[int, int]).parse(d)
    assert excinfo.match("Setting 'val' \(value: 'a', type: str\)")
    assert excinfo.match('does not have the right type')
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', Dict[str, Union[int, complex]]).parse({'a': 'b'})
    assert excinfo.match("Setting 'val' \(value: 'b', type: str\)")
    assert excinfo.match('does not have the right type')
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Value('val', Dict[str, int]).parse(None)
    assert excinfo.match("Setting 'val' \(value: None, type: NoneType\)")
    assert excinfo.match('does not have the right type')
    assert excinfo.type == ValueError


def test_nested_dicts():
    '''Test the parsing of nested dictionaries.'''
    d = {'a': {('b', 'c'): 4}}
    assert Value('val', Dict[str, Dict[Tuple[str], int]]).parse(d) == {'a': {('b', 'c'): 4}}
    assert Value('val', Dict[str, Dict[Tuple[str], str]]).parse(d) == {'a': {('b', 'c'): '4'}}


def test_wrong_generics():
    '''Test that using a generic without arguments fails'''
    with pytest.raises(ValueError) as excinfo:
        Value('val', List).parse([1,2,3])
    assert excinfo.match("Invalid requested type \(List\), generic types must contain arguments.")
    assert excinfo.type == ValueError
